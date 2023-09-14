from app.backend import VideoSource, VideoPlayer, SessionProfile
from app.pipeline import EmotionPerceptionTool

""" Video.record = escribe el video procesado"""

def video_inference(video_path):
    camera = VideoSource(video_path)  # '/media/lecun/HD/Grimmat/Emotions Video/BIOMEDICINA INTELIGENTE MEMORIAS.mp4')
    pipeline = EmotionPerceptionTool()
    session = SessionProfile('/media/lecun/HD/Grimmat/Emotions Video/code/IOEPT/sessions/')
    size = None
    video = VideoPlayer(size, camera, pipeline)  # , session)
    video.record(name=f'{video_path.rsplit(".")[0]}_processed.mp4', show=False)
    return f'{video_path.rsplit(".")[0]}_processed.mp4'


if __name__ == '__main__':

    """pipeline = instancio pipeline (utilizo percepcion de emociones y cargo el modelo) - se dice qué procesamiento se le va a hacer al video
        camera = Cargo el video source (camara o video) - VideoSource (es un CD) encapsula un video para luego cargarlo y procesarlo
        session = Se crea una sesión en la carpeta \session
        videoPlayer = coge el source y el pipeline y empieza a correr - reproduce el video (es un DVD que ejecuta el CD)
    """
    
    pipeline = EmotionPerceptionTool('IOEPT_v0.1.pt')  
    camera = VideoSource(0) # path to video or 0 to camera  
    session = SessionProfile('/sessions')   
    size = None 
    video = VideoPlayer(size, camera, pipeline)#, session)  
    video.run()