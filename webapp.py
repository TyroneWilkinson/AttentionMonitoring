import streamlit as st
import cv2 as cv
import tempfile
import numpy as np
from tensorflow.keras.models import load_model

st.title("Attention Monitoring")
st.write('''Fine-tuned the top layers of a pre-trained ResNet-12 model to
accurately classify images, videos and live feed.''')
st.write('''See blog for more information: 
https://nycdatascience.com/blog/student-works/attention-monitoring/''') 

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def model_loader(location):
    # location: './resnet152v2/model2'
    return load_model(location)

def file_checker(vid_file):  
    cap = cv.VideoCapture(vid_file)
    if not cap.isOpened():
        st.write("Cannot open camera")
        exit()
        return None
    return cap

def video_labeler(vid_file, vid_name, model):
    writer = None
    frame_width = frame_height = None
    classes = ['ATTENTIVE', 'NOT ATTENTIVE']
    VIDEO_OUT = "videos/out/"
    VIDEO_NAME=vid_name
    stframe = st.empty()
    count = 0
    # Loop over frames from the video stream
    while True:
        # Capture frame-by-frame
        is_present, frame = vid_file.read()

        # If is_present is false, frame was read incorrectly or we have 
        # reached the end of the video
        if not is_present:
            st.write("Video classification is complete.")
            break

        # Get frame dimensions if empty
        if frame_width is None or frame_height is None:
            frame_width, frame_height = frame.shape[:2]

        # Operations on the frame: Convert, Resize, Rescale
        output = frame.copy()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (224, 224)).astype("float32")
        frame /= 255

        # Label the current frame
        percent_pred = model.predict(np.expand_dims(frame, axis=0))[0]
        percent_pred = float(*percent_pred)*100     
        preds = 0 if percent_pred < 99 else 1
        label = classes[preds]    
        

        # Write the label on the output frame
        text = f"{label}"
        org = (35, 50)
        font = cv.FONT_HERSHEY_DUPLEX
        fontScale = 1.25
        color = (0, 255, 0)
        thickness = 3
        cv.putText(output, text, org, font, fontScale, color, thickness)

        # Check if videowriter is None
        if writer is None:
            # Define the codec and create VideoWriter object
            fourcc = cv.VideoWriter_fourcc(*"DIVX")
            writer = cv.VideoWriter(VIDEO_OUT+VIDEO_NAME, fourcc, 30,
                (frame_width, frame_height), True)

        # Show output images
        cv.imshow("Output", output)
        stframe.image(output)
        
        # Captures every 15th frame (for speed)
        count += 15
        vid_file.set(cv.CAP_PROP_POS_FRAMES, count)

    # Rrelease the file pointers
    vid_file.release()
    writer.release()

    # Close all windows
    cv.destroyAllWindows()

vid_file = st.file_uploader("Upload the video you want labeled.","mp4")
model_dir = './resnet152v2/model2'

if vid_file:
    st.video(vid_file)
    vid_cv = tempfile.NamedTemporaryFile(delete=False)
    vid_cv.write(vid_file.read())
    vid_name = vid_cv.name
    vid_cv = cv.VideoCapture(vid_cv.name)
    model=model_loader(model_dir)
    video_labeler(vid_cv, vid_name, model)
