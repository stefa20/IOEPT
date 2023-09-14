import cv2
from app.opencv_utils import resize_image, convert_color_space, BGR2RGB, show_image
import numpy as np
import pandas as pd
import os
from datetime import datetime
from app.pipeline import FER_LABELS

class VideoSource(object):
    """Video Source abstract class - envuelve el video para poder sacarle fotos al video
    By default opens the web camera using openCV functionality.
    It can be inherited to overwrite methods in case another camera API exists.
    Permite conectarse con webcam o leer un video que ya esté

     # Properties
        source_id: Int or Str. Integer with id of connected camera or path to video file in correct format.
        pipeline: Function. Should take RGB image as input and it should
            output a dictionary with key 'image' containing a visualization
            of the inferences. Built-in pipelines can be found in
            ``paz/processing/pipelines``.
    # Methods
        start()
        stop()
        read()
    """
    def __init__(self, source_id=0):
        # TODO load parameters from camera name. Use ``load`` method.
        self.source_id = source_id  # El source_id puede ser la cámara o el path del video
        self.source = None
        self.intrinsics = None
        self.distortion = None

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, intrinsics):
        self._intrinsics = intrinsics

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, distortion):
        self._distortion = distortion

    def start(self):    
        """ Starts capturing device - se le dice al video que vamos a comenzar 
        # Returns
            Camera object.
        """
        self.source = cv2.VideoCapture(self.source_id)  #VideoCapture para conectarse a la cámara con el source_id que se seleccionó    
        if self.source is None or not self.source.isOpened():
            raise ValueError('Unable to open device', self.source_id)   #Error cuando la cámara ya está abierta
        return self.source

    def stop(self):
        """ Stops capturing device - permite liberar la cámara
        """
        return self.source.release()

    def read(self):
        """Reads camera input and returns a frame - permite obtener una imagen
            le dice a la cámara que le dé una imagen
            se obtienen varias imágenes por segundo
        # Returns
            Image array.
        """
        frame = self.source.read()[1]
        return frame

    def is_open(self):
        """Checks if camera is open
        # Returns
            Boolean
        """
        return self.source.isOpened()

    def get_fps(self):
        """"Returns framse per second of the source passed
        # Returns
            Int
        """
        return round(self.source.get(cv2.CAP_PROP_FPS))

    def calibrate(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

class VideoPlayer(object):
    """Performs visualization inferences in a real-time video - toma el VideoSource y le pide fotos
    # Properties
        image_size: List of two integers. Output size of the displayed image.
        source: VideoSource class pre-initialized.
        pipeline: Function. Should take RGB image as input and it should
            output a dictionary with key 'image' containing a visualization
            of the inferences. Built-in pipelines can be found in
            ``paz/processing/pipelines``.

    # Methods
        run()
        record()
    """
    "TODO: Change image_size position"
    def __init__(self, image_size, source, pipeline=None, session=None):
        self.image_size = image_size    #Inicializo el tamaño de la imagen
        self.video = source #Indica cuál es el video que es un source - se pasa la instancia de VideoSource 

        if session is not None: 
            self.profiles = session #Session donde se guarda los perfiles de los usuarios

        if pipeline is not None:
            self.pipeline = pipeline    #El pipeline es si deseo realizar un tipo de procesamiento al video
                                        #Si no deseo el procesamiento simplemente no se pasa el pipeline y se reproduce el video como si fuera un reproductor

    def start(self):
        self.video.start()  #Se inicia el video, es el mismo start que está en la clase de VideoSource
        return

    def step(self):
        """ Runs the pipeline process once
        # Returns
            Inferences from ``pipeline``.
        """
        #Revisa si la cámara está abierta, si no está abierta se ejecuta el start
        if self.video.is_open() is False:
            raise ValueError('Camera has not started. Call ``start`` method.')  

        frame = self.video.read()   #Cuando la cámara está abierta se le pide un frame al video
        if frame is None:
            print('Frame: None')
            return None
        #print(f'Original Shape: {frame.shape}')

        # all pipelines start with an RGB image
        if hasattr(self, 'pipeline'):   #Use pipeline if passed
            pipe_out = self.pipeline(frame)  # Get results from pipeline - se pasa el frame al pipeline si tengo pipeline

            if hasattr(self, 'profiles'):
                self.profiles.update(pipe_out)  # save records updating profiles - 
                                                # si tengo perfiles paso el resultado del pipeline a los perfiles para que lo graben

            return pipe_out

        return {'image': frame} #Retorna la imagen - el frame 

    def run(self, display=True, writer= None):
        """Opens camera and starts continuous inference using ``pipeline``,
        until the user presses ``q`` inside the opened window.
        """
        self.start()    #Inicia todo el proceso (abra la cámara, inicie el video)
        print('video start processing.')

        if hasattr(self, 'profiles'):
            self.profiles.update_session_profile('fps', self.video.get_fps())   #Se inicia el perfil

            source_name = self.video.source_id
            if source_name == 0:
                source_name = "webcam"  #Si es webcam se escribe en el perfil que es una webcam
            else:
                source_name = source_name.rsplit('/', 2)[-1]

            self.profiles.update_session_profile('video_source',
                                                 source_name)   #Si no es webcam escribe el nombre del video
        empty_frames = 0
        th = 30
        while True:
            output = self.step()    #Un paso es obtener la imagen, hacer procesamiento, obtener las emociones

            if output is None and empty_frames > th:
                print('---- Frames completed, process finished ----')   #Si no hay más frames termina el proceso
                break
            elif output is None and empty_frames <= th: #Si hay frames vacíos simplemente se ignoran o partes del video donde no hay rostros
                empty_frames = empty_frames + 1
                print(empty_frames)
                continue

            empty_frames = 0
            if self.image_size is not None:
                image = resize_image(output['image'], tuple(self.image_size))
            else:
                image = output['image']

            if display:
                show_image(image, 'inference', wait=False)  #Muestra la imagen durante el procesamiento

            if writer is not None:
                writer.write(convert_color_space(image, BGR2RGB))   #Se escribe la imagen

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video.stop()
        cv2.destroyAllWindows()

        if hasattr(self, 'profiles'):
            self.profiles.save_session_profiles()

    def record(self, name='./video.avi', show=False, fps=20, fourCC='MP4V'):
        """Opens camera and records continuous inference using ``pipeline``.
        # Arguments
            name: String. Video name. Must include the postfix .avi.
            fps: Int. Frames per second.
            fourCC: String. Indicates the four character code of the video.
            e.g. XVID, MJPG, X264.
        """
        self.video.start()  #Comienza el video
        fps = self.video.get_fps()  #Se obtienen los frames por segundo (fps) porque son importantes para esribir

        if self.image_size is None:
            test_frame = self.video.source.read()[1]
            h, w = test_frame.shape[:2]
            self.image_size = (w, h)

        # print(size)
        #Clases que permiten escribir videos
        fourCC = cv2.VideoWriter_fourcc(*fourCC)
        writer = cv2.VideoWriter(name, fourCC, fps, self.image_size)
        self.run(writer=writer, display=show)
        # while True:
        #     output = self.step()
        #     if output is None:
        #         continue
        #     image = resize_image(output['image'], tuple(self.image_size))
        #     show_image(image, 'inference', wait=False)
        #     writer.write(image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # self.stop()
        writer.release()
        cv2.destroyAllWindows()

class SessionProfile:
    """ A Class to record Inference information in a session.
    For functionality it need VideoPlayer to start with a clear view of the faces to create
    profiles and perform record.

    methods:
        update
            Args: data. Dict: with keys boxes(boundingBox detection)
            and data(dictionary with 'probs' and other inference related data.

        create_profile()
        sum_frame()
        save_session_profiles()

    """
    def __init__(self, save_path='./sessions/'):

        self.user_profiles: [UserProfile] = []
        self.frame: int = 0
        self.path = save_path
        self.session_profile: dict = {}

    def update(self, data):
        self.sum_frame()
        if len(self.user_profiles) == 0:
            self.create_user_profile(data)
        self.update_user_profiles(data)

        # if 'detector' in data:
        #     self.update_session_profile('face count', data['detector']['face_count'])

    def update_session_profile(self, key, item):
        self.session_profile[key] = [item]

    def update_user_profiles(self, data):
        if 'data' not in data: # No detection from pipeline in frame
            for profile in self.user_profiles:
                profile.update(np.zeros((7,)))
            return

        probs = data['data']['probs']
        for data, profile in zip(probs, self.user_profiles):
            profile.update(data)

    def sum_frame(self):
        self.frame += 1

    def create_user_profile(self, data):
        if 'detector' not in data:  # Ignore creation of profiles as not faces are recognized in the videoo
            return
        boxes = data['detector']['boxes']
        self.user_profiles = [UserProfile(f'user{i}', current_bbox=box) for i, box in enumerate(boxes)]
        self.update_session_profile('start_frame', self.frame)
        self.update_session_profile('user_count', len(boxes))
        return

    def save_session_profiles(self):

        sess_path = datetime.now().strftime("%D_%H:%M:%S").replace('/','-')
        self.sess_name = sess_path
        save_path = os.path.join(self.path, sess_path)

        if not os.path.exists(self.path):
            print("sessions path created in project root folder")
            os.mkdir(self.path)

        os.mkdir(save_path)

        sess_profile = pd.DataFrame(self.session_profile)
        sess_profile.to_csv(f'{save_path}/{sess_path}_info.csv', index=False, encoding='UTF-8')

        for profile in self.user_profiles:
            results = profile.get_profile_results()
            df = pd.DataFrame(np.concatenate([results,]), columns=FER_LABELS)
            df.to_csv(f'{self.path}/{sess_path}/{profile.user_id}_results.csv', index=False, encoding='UTF-8')

class UserProfile(object):
    def __init__(self, user_id, current_bbox):
        self.user_id = user_id
        self.user_pos = current_bbox
        self.user_results = []

    def update(self, instance):
        self.user_results.append(instance)

    def get_profile_results(self):
        return self.user_results

    def check_position(self, bbox, instance):
        raise NotImplementedError

if __name__ == '__main__':

    camera = VideoSource()

    video = VideoPlayer((800,600), camera)

    video.record(name="./test.mp4", show=False, size=(800,600))