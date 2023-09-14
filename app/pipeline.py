from facenet_pytorch import MTCNN, extract_face
from torch import nn, load, no_grad
from app.opencv_utils import convert_color_space, RGB2GRAY, BGR2RGB
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torch import sigmoid
import torch

#Los LABELS dependen del entrenamiento del modelo
FER_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class FaceDetector:
    """" Face Detection pipeline
          Arguments:
              min_face_size: Int. Minimun pixel size of detected faces
              device: Str. cuda or cpu.
            init = se inicializan las clases y demás cosas que se van a utilizar
            self.mtcnn = se instancia la clase para utilizarla en las demás clases como una función
          """
    def __init__(self, min_face_size=90, device='cuda'):
        self.mtcnn = MTCNN(keep_all=True, device=device,
                           image_size=48, min_face_size=min_face_size)  

    def __call__(self, frame):
        """" Perform face detection on a np.Array frame
        Arguments:
            frame: np.Array, should be RGB image - una imagen en Array y en formato RGB

        Return:
            Dict: image: np.Array. processed RGB frame with faces detected (devulve un diccionario con la imagen del rostro recortada)
                boxes: list of bounding box coordinates of faces (recuadros donde está la imagen)
                faces: Tensor. cropped faces as gray scale tensor batch.

        call = cuando se ejecuta la clase - se llama utilizando FaceDetector(Frame)
        DetectorOut = Me devuelve un rostro

        boxes = se aplica la función detect para detectar los boxes
        faces = extraigo los rostros 
        """
        face_count = 0
        boxes, _ = self.mtcnn.detect(frame) 
        gray = convert_color_space(frame, RGB2GRAY)
        faces = self.mtcnn.extract(gray, boxes, save_path=None)

        # Draw faces in frames for plotting
        #Dibuja el recuadro en la imagen grande
        frame_draw = Image.fromarray(frame.copy())  #Copia la imagen original
        draw = ImageDraw.Draw(frame_draw)   #Se le hace un dibujo
        if boxes is not None:
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(97, 198, 255), width=5)   #Se ubican los rectángulos sobre los rostros
                face_count = len(boxes)

        pos_frame = np.asarray(frame_draw)  #Lo convierto en un numpy array
        #Retorna la imagen grande con los recuadros sobre los rostros, las coordenadas de los recuadros, los rostros recortados y el número de rostros detectados
        return {'image': pos_frame, 'boxes': boxes, 'faces': faces, 'face_count': face_count}


class InstanceEmotionPercepTool:
    """" Emotion perception on cropped faces images.
             Arguments:
                 artifact_path: Str. path to trained model - path del modelo entrenado
                 arq: nn.Module. Model definition to load state_dict (default=None)
             """
    def __init__(self, artifact_path, arq=None, device='cpu'):
        super(InstanceEmotionPercepTool, self).__init__()
        self.device = device    #Asigno el tipo de dispositivo
        self.emotion_labels = FER_LABELS    #Asigno los LABELS
        if arq is not None:
            self.model = self.load_model(artifact_path, arq)   #Se carga el modelo   
        else:
            self.model = load(artifact_path)

    #Función para cargar el modelo entrenado .pt
    def load_model(self, artifact_path, arq):
        model = arq
        checkpoint = load(artifact_path)  # , map_location=device)
        state_dict = checkpoint['net']
        model.load_state_dict(state_dict)
        return model

    def __call__(self, imgs):
        """" Perform emotion classification on a batch of faces images Tensor.
        Arguments:
            imgs: Tensor. tensor of images with faces - recibe una imagen en un formato tensor 

        Returns:
            Dict. probs: Tensor. with the output of the model as the activation of last neurons.
            emotions: list[Str]. list with max expression label of each passed face.
        """
        self.model.eval()
        self.model.to(self.device)
        with no_grad():
            probs = self.model(imgs.to(self.device))    #Se pasa la imagen al modelo y saca las probabilidades con una función sigmoide
            probs = sigmoid(probs)
        max_prob_ind = probs.argmax(axis=-1)
        emotions = [self.emotion_labels[ind] for ind in max_prob_ind]   #Se obtienen las emociones

        if torch.cuda.is_available():
            probs = probs.cpu()
        #Retorna la lista de las 7 probabilidades para las 7 emociones
        return {'probs': probs.numpy(), 'emotions': emotions}


class EmotionPerceptionTool:
    """
    This class implementes FaceDetector to crop faces and passforward emotion detection network to return
    emotional profiles of each faces.
    """

    def __init__(self, model_path='./app/IOEPT_v0.1.pt'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.face_detector = FaceDetector(device=device)    #Inicializo el detector de rostros
        self.emotion_detector = InstanceEmotionPercepTool(model_path, device=device)    #Inicializo el detector de emociones

    def __call__(self, frame):
        """
        Args: frame, image array as loaded with cv2.imread
        """
        frame = convert_color_space(frame, BGR2RGB)
        detector_out = self.face_detector(frame)    #Detecta los rostros 

        if detector_out['faces'] is None:
            return {'image': detector_out['image']} #Extraigo las caras (si no hay caras le digo que retorne la misma imagen)

        # Extract faces and detect emotion
        emotion_data = self.emotion_detector(detector_out['faces']) #Se detectan las emociones a las caras 

        # Draw faces in frames for plotting
        frame_draw = Image.fromarray(detector_out['image'].copy())  
        draw = ImageDraw.Draw(frame_draw)
        # try:
        fnt = ImageFont.truetype("app/FreeMonoBold.ttf", 35)
        # except OSError:
        #     fnt = ImageFont.load_default()
        if detector_out['boxes'] is not None:
            for box, label in zip(detector_out['boxes'], emotion_data['emotions']):
                boxx = box.tolist()[:2]
                boxxx = [boxx[0], boxx[1]-10]
                draw.text(boxxx, label, fill=(97, 198, 255), width=24, font=fnt)

        pos_frame = np.asarray(frame_draw)
        #Retorna la imagen grande con los recuadros de los rostros dibujados y el nombre de la emoción
        #se devuelven las coordenadas de los recuadros y devuelve las probabilidades y las emociones
        return {'image': pos_frame, 'detector': detector_out, 'data': emotion_data}


if __name__ == '__main__':

    import cv2
    import matplotlib.pyplot as plt

    img = cv2.imread("/home/lecun/Downloads/test_img.jpg")  #Se carga un rostro usando imread

    pipeline = EmotionPerceptionTool()  #Instancio la clase
    result = pipeline(img)  #Paso la imagen al pipeline para que él dé los resultados

    print(result)
    plt.imshow(result['image'])
    plt.show()