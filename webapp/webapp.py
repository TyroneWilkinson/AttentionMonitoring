import streamlit as st
import cv2 as cv
import tempfile
import numpy as np
import gdown
from tensorflow.keras.models import load_model

st.title("Attention Monitoring")
st.write('''Hi! I'm Tyrone! I fine-tuned the top layers of a pre-trained ResNet-152 model
in order to accurately classify images, videos and live feed.
"Classifying for what exactly...?" you might ask. Whether or not you are paying attention.''')
st.write('''See my blog for more information: 
https://nycdatascience.com/blog/student-works/attention-monitoring/''') 

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def model_loader(model_location):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    _ = gdown.download(model_location,tmp)
    return load_model(tmp.name)

def video_labeler(vid_file, vid_name, model, frame_num):
    frame_width = frame_height = None
    classes = ['ATTENTIVE', 'NOT ATTENTIVE']
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

        # Show output images
        stframe.image(output, channels='BGR')
        
        # Captures every 15th frame (for speed)
        count += frame_num
        vid_file.set(cv.CAP_PROP_POS_FRAMES, count)

    # Release the file pointers
    vid_file.release()

vid_file = st.file_uploader("Upload the video you want classified.","mp4")
model_location = "https://drive.google.com/uc?id=1gUsVPU65Dd-DGoOgF-lwW9w_AEja1OE_&export=download"

if vid_file:
    st.video(vid_file)
    frame_num = st.number_input(label='''The model will only label every nth frame.
    Enter a value between 1 and 15 inclusive (I would start with 15).''',
                             min_value=0,
                             max_value=15,
                             value=0)
    
    if frame_num > 0 and frame_num <= 15:
        vid_cv = tempfile.NamedTemporaryFile(delete=False)
        vid_cv.write(vid_file.read())
        vid_name = vid_cv.name
        vid_cv = cv.VideoCapture(vid_cv.name)
        model=model_loader(model_location)
        video_labeler(vid_cv, vid_name, model, frame_num)
        
        if st.button("Click to rerun classification (or change frame value above to rerun)"):
            st.script_request_queue.RerunData(None)



