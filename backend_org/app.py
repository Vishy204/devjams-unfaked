import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Wrapper class for model
#Image dimensions standardisation
image_dimensions={'height':256,'width':256,'channels':3}

class Classifier:
    def __init__():
        self.model = 0

    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
  #Initiialising the class
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    def init_model(self):
      #Input Layer
        x = Input(shape = (image_dimensions['height'], image_dimensions['width'], image_dimensions['channels']))

#4 convolutional blocks
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)

#Initializing model and loading weights
model=Meso4()
model.load("model_weights/Meso4_DF (1).h5")
import cv2
import numpy as np
from typing import List

# Function to preprocess the image
def preprocess_image(image, target_size=(256, 256)):
    # Convert BGR to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match model input
    img = cv2.resize(img, target_size)

    # Normalize the image
    img = img.astype('float32') / 255.0

    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img

# Function to analyze the video and make predictions
def analyze_video(video_path, model):
    sum_predictions = 0
    list_of_predictions = []

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_image(frame)

        # Make predictions
        predictions = model.predict(preprocessed_frame)

        # Display predictions on the frame
        prediction_value = float(predictions[0][0])
        list_of_predictions.append(prediction_value)


        # Display the frame
        # cv2.imshow('Video', frame)  # Uncomment to display video with prediction

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Calculate the average of the predictions
    for i in list_of_predictions:
        sum_predictions+=i
   # Check if any predictions were made
    if len(list_of_predictions) > 0:
        average_prediction = sum_predictions / len(list_of_predictions)
    else:
        average_prediction = 0  # or handle this case as you prefer
    

    # Determine if the video is real or fake based on the average prediction
    return average_prediction,list(list_of_predictions)



from moviepy.editor import VideoFileClip
import speech_recognition as sr
import os

def convert_video_to_wav(video_file, output_audio="extracted_audio.wav"):
    """
    Convert a video file to .wav audio format using moviepy.
    """
    try:
        # Load the video file
        video = VideoFileClip(video_file)

        # Extract audio and write it to a .wav file
        audio = video.audio
        audio.write_audiofile(output_audio, codec='pcm_s16le')  # 'pcm_s16le' for .wav format
        print(f"Audio extracted and saved as {output_audio}")
        return output_audio
    except Exception as e:
        print(f"Error converting video to audio: {e}")
        return None

def audio_to_text(audio_file):
    """
    Convert an audio file (wav format) to text using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()

    try:
        # Load the audio file
        with sr.AudioFile(audio_file) as source:
            print("Recognizing audio...")
            audio_data = recognizer.record(source)

            # Convert speech to text using Google's Speech-to-Text
            text = recognizer.recognize_google(audio_data)
            print("Transcribed Text: ", text)
            return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Error with Google Speech Recognition service: {e}")
        return None

def video_to_text(video_file):
    """
    Complete process: Convert a video file to text by extracting audio and
    applying speech-to-text conversion.
    """
    # Step 1: Convert the video to wav format
    audio_file = convert_video_to_wav(video_file)

    if audio_file:
        # Step 2: Convert the extracted audio to text
        text = audio_to_text(audio_file)
        return text
    else:
        print("Audio extraction failed.")
        return None
import cv2
import dlib
import tensorflow as tf
from typing import List
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import os
import cv2
from keras.models import Model
from keras.layers import SpatialDropout3D, Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, Bidirectional, LSTM, Dense, TimeDistributed, ZeroPadding3D, Flatten, Dropout, GRU
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
a=""
# Load dLib's pre-trained face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model_weights/shape_predictor_68_face_landmarks (1).dat")

def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    target_size = (60, 40)  # Define a fixed size for the cropped lip region (height, width)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale (dLib expects grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector(gray_frame)

        for face in faces:
            # Get the landmarks/parts for the face
            landmarks = predictor(gray_frame, face)

            # Extract the coordinates of the lips (points 48-67)
            lip_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)]

            # Create a bounding box around the lips
            x_coords, y_coords = zip(*lip_points)
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Crop the lip region from the frame
            lip_region = frame[min_y:max_y, min_x:max_x]

            # Resize the lip region to the target size
            lip_region_resized = cv2.resize(lip_region, target_size)

            # Convert to grayscale and append to frames
            lip_region_gray = tf.image.rgb_to_grayscale(lip_region_resized)
            frames.append(lip_region_gray)

    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_data(path: str):
    path = bytes.decode(path.numpy())
    frames = load_video(path)

    return frames

def mappable_function(path:str) ->List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


inputs = Input(shape=(75, 40, 60, 1))

    # Convolutional layers with BatchNormalization and SpatialDropout3D
x = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(inputs)
x = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(x)
x = Activation('relu', name='actv1')(x)
x = BatchNormalization(name='batc1')(x)
x = SpatialDropout3D(0.5, name='spatial_dropout3d_1')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(x)
x = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(x)
x = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(x)
x = Activation('relu', name='actv2')(x)
x = BatchNormalization(name='batc2')(x)
x = SpatialDropout3D(0.5, name='spatial_dropout3d_2')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(x)
x = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(x)
x = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(x)
x = Activation('relu', name='actv3')(x)    
x = BatchNormalization(name='batc3')(x)
x = SpatialDropout3D(0.5, name='spatial_dropout3d_3')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(x)

    # Reshape for RNN layers
x = TimeDistributed(Reshape((-1,)), name='time_distributed_1')(x)
    # RNN layers
x = Bidirectional(GRU(256, return_sequences=True, kernel_initializer=tf.keras.initializers.Orthogonal  , name='gru1'), merge_mode='concat')(x)
x = Bidirectional(GRU(256, return_sequences=True, kernel_initializer=tf.keras.initializers.Orthogonal , name='gru2'), merge_mode='concat')(x)

    # Dense and Activation layers
x = Dense(41, kernel_initializer='he_normal', name='dense1')(x)
x = Activation('softmax', name='softmax')(x)

    # Define the model
model_lip = tf.keras.Model(inputs, x)

model_lip.load_weights("../model_weights/dlib3_lipnet_model (1).h5")
model_lip.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# Example model prediction line
def model_predict(sample):
    # Assuming your model expects input of shape (1, 75, 40, 60, 1)
    yhat = model_lip.predict(tf.expand_dims(sample, axis=0))
    return yhat

# Function to split video into chunks of (75, 40, 60, 1)
def split_video_into_chunks(video, chunk_size=75):
    """
    Splits the video into chunks of shape (75, 40, 60, 1).
    If the last chunk has fewer than 75 frames, it will be padded with zeros.
    """
    chunks = []
    total_frames = video.shape[0]  # The first dimension is the number of frames (x)

    # Iterate through the video in steps of 75 frames
    for i in range(0, total_frames, chunk_size):
        chunk = video[i:i + chunk_size]
        
        # If the chunk has fewer than 75 frames, pad it
        if chunk.shape[0] < chunk_size:
            padding = np.zeros((chunk_size - chunk.shape[0], 40, 60, 1))
            chunk = np.concatenate((chunk, padding), axis=0)
        
        chunks.append(chunk)
    
    return np.array(chunks)

# Function to process video through the model
def process_video_through_model(video):
    """
    Takes a video of shape (x, 40, 60, 1) where x is the number of frames, splits it into
    chunks of shape (75, 40, 60, 1), and runs each chunk through the model.
    """
    # Split the video into chunks of (75, 40, 60, 1)
    video_chunks = split_video_into_chunks(video)
    
    predictions = []
    
    # Run each chunk through the model
    for chunk in video_chunks:
        print(chunk.shape)
        yhat = model_lip.predict(tf.expand_dims(chunk, axis=0))
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        tensor = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
        string_value = tensor[0].numpy().decode("utf-8")
        predictions.append(string_value)
    
    # Concatenate all predictions into a single string
    full_prediction_string = ' '.join(predictions)
    
    return full_prediction_string





import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import base64
import io
from PIL import Image

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to track eyelid movement for irregularities
def analyze_blink_smoothness(ear_values, threshold=0.05):
    differences = np.diff(ear_values)  # Get the differences between consecutive EAR values
    irregular_movements = sum(abs(diff) > threshold for diff in differences)  # Count large differences
    return irregular_movements

# Main function to process the video and return results
def process_video(video_path):
    # Load the pre-trained facial landmark detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model_weights/shape_predictor_68_face_landmarks (1).dat")

    # Eye aspect ratio threshold for blink detection
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3
    BLINK_IRREGULARITY_THRESH = 3  # Customize based on experiments

    # Initialize counters and variables
    blink_counter = 0
    total_blinks = 0
    irregular_blinks = 0
    blink_started = False
    ear_values = []

    # Start video capture
    video_capture = cv2.VideoCapture(video_path)

    # Grab the indexes of the facial landmarks for the left and right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    frame_to_return = None  # Initialize the frame to return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            # Visualize the landmarks for the eyes by drawing circles
            for (x, y) in leftEye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw circles for the left eye
            for (x, y) in rightEye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw circles for the right eye

            if ear < EYE_AR_THRESH:
                blink_counter += 1
                ear_values.append(ear)  # Track EAR during blink
                if not blink_started:
                    blink_started = True
                    ear_values = [ear]  # Start tracking from the first frame of blink
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    total_blinks += 1

                    # Check for irregularities in blink smoothness
                    irregularities = analyze_blink_smoothness(ear_values)
                    if irregularities > BLINK_IRREGULARITY_THRESH:
                        irregular_blinks += 1

                blink_counter = 0
                blink_started = False

        # Store the current frame for returning later
        frame_to_return = frame

    video_capture.release()

    # If a frame was processed, convert it to base64
    if frame_to_return is not None:
        # Add text for total blinks and irregular blinks to the frame
        cv2.putText(frame_to_return, f"Total Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_to_return, f"Irregular Blinks: {irregular_blinks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convert the frame to a PIL image and then to a byte buffer
        _, buffer = cv2.imencode('.png', frame_to_return)
        pil_image = Image.open(io.BytesIO(buffer))
        
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        # Encode the image as base64 for return or transmission
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # Return the base64 image and blink information
        return {"image_base64": img_base64, "total_blinks": total_blinks, "irregular_blinks": irregular_blinks}
    else:
        # Return default values if no frame was processed
        return {"image_base64": None, "total_blinks": total_blinks, "irregular_blinks": irregular_blinks}


import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Image dimensions standardisation
image_dimensions={'height':256,'width':256,'channels':3}

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)
   

def analyze_audio(input_audio_path):
    scaler_filename = "model_weights/scaler (1).pkl"
    model_filename = "model_weights/svm_model_best (1).pkl"
    svm_classifier = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")
        return
    elif not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")
        return

    mfcc_features = extract_mfcc_features(input_audio_path)

    if mfcc_features is not None:
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
        prediction = svm_classifier.predict(mfcc_features_scaled)
        if prediction[0] == 0:
            return "The input audio is classified as genuine."
        else:
            return "The input audio is classified as deepfake." 
    else:
        return "Error: Unable to process the input audio."

def main():
    
    # Check if each class has at least two samples
    if len(X_genuine) < 2 or len(X_deepfake) < 2:
        print("Each class should have at least two samples for stratified splitting.")
        print("Combining both classes into one for training.")
        X = np.vstack((X_genuine, X_deepfake))
        y = np.hstack((y_genuine, y_deepfake))
    else:
        X = np.vstack((X_genuine, X_deepfake))
        y = np.hstack((y_genuine, y_deepfake))



from fastapi import FastAPI, File, UploadFile
import os
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import nest_asyncio
import cv2
import numpy as np
from fuzzywuzzy import fuzz


nest_asyncio.apply()

app = FastAPI()
UPLOAD_DIRECTORY = "backend/uploaded_files"

# Ensure the directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(video: UploadFile = File(None), audio: UploadFile = File(None)):
    result = None
    random_array = []
    micro=None
    frame_base64=None
    dct_base64=None
    image_base64=None
    total_blinks=None
    irregular_blinks=None
    full_prediction_string=None
    transcribed_text = None
    similarity=None
    gaze=None
    lip=None
    mfcc1=None
    mfcc2=None
    mfcc3=None
    mfcc1_64=None
    mfcc2_64=None
    prediction=None
    mfcc3_64=None
    final_result=None
    result1=None
 
   
    
    if video:
        video_path = os.path.join(UPLOAD_DIRECTORY, video.filename)
        with open(video_path, "wb") as buffer:
            buffer.write(await video.read())

        # Call the video analysis function
     
        result, random_array = analyze_video(video_path,model)
        if result>0.5:
            micro="The Microexpressions are human-like and this is not a deepfake."
        else:
            micro="The Microexpressions are not human-like and this is a deepfake."

       

        eye_result=process_video(video_path)
        image_base64 = eye_result["image_base64"]
        total_blinks = eye_result["total_blinks"]
        irregular_blinks = eye_result["irregular_blinks"]

        if irregular_blinks>1:
            gaze="The blinks are irregular which suggest that it is a deepfake."
        else:
            gaze="The blinks are regular which suggests that it is not a deepfake"

        video = load_data(tf.convert_to_tensor(video_path))
        full_prediction_string = process_video_through_model(video)

        transcribed_text = video_to_text(video_path)
        similarity = fuzz.ratio(full_prediction_string, transcribed_text)
        if similarity>50:
            lip="The lip-synchnorisation Score is high enough for it to be not a deepfake."
        elif similarity==0:
            lip="There is no audio present"
        else:
            lip="The lip-synchronisation Score is low so it is deepfake"

        if (similarity<60 and similarity!=0) or result<0.5 or irregular_blinks>1:
            final_result="Final Conclusion:This is a deepfake."
        else:
            final_result="Final Conclusion:This is not a deepfake"

    if audio:
        audio_path = os.path.join(UPLOAD_DIRECTORY, audio.filename)
        with open(audio_path, "wb") as buffer:
            buffer.write(await audio.read())
        # Create in-memory buffers for images
         # Load audio and plot data
        real_ad, real_sr = librosa.load(audio_path)

        # Create in-memory buffers for images
        mfcc1 = io.BytesIO()
        mfcc2 = io.BytesIO()
        mfcc3 = io.BytesIO()

        # Plot waveform and save as image
        plt.figure(figsize=(12, 4))
        plt.plot(real_ad)
        plt.title("Audio Data")
        plt.tight_layout()  # Ensure the layout is tight to avoid clipping
        plt.savefig(mfcc1, format='png', bbox_inches='tight')  # Save plot to buffer
        plt.close()  # Close the figure
        mfcc1.seek(0)  # Reset buffer position

        # Plot Mel spectrogram
        real_mel_spect = librosa.feature.melspectrogram(y=real_ad, sr=real_sr)
        real_mel_spect = librosa.power_to_db(real_mel_spect, ref=np.max)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(real_mel_spect, y_axis="mel", x_axis="time")
        plt.title("Audio Mel Spectrogram")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.savefig(mfcc2, format='png', bbox_inches='tight')
        plt.close()  # Close the figure
        mfcc2.seek(0)

        # Plot MFCCs
        real_mfccs = librosa.feature.mfcc(y=real_ad, sr=real_sr)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(real_mfccs, sr=real_sr, x_axis="time")
        plt.colorbar()
        plt.title("Audio MFCCs")
        plt.tight_layout()
        plt.savefig(mfcc3, format='png', bbox_inches='tight')
        plt.close()  # Close the figure
        mfcc3.seek(0)

        # Convert to base64 strings
        mfcc1_64 = base64.b64encode(mfcc1.read()).decode('utf-8')
        mfcc2_64 = base64.b64encode(mfcc2.read()).decode('utf-8')
        mfcc3_64 = base64.b64encode(mfcc3.read()).decode('utf-8')

        
       
        

      

       
        result1 = analyze_audio(audio_path)  # Single value result for audio

            
 

        return {
            "result": result,
            "random_array": random_array,
            "prediction": prediction,
            "frame_base64": frame_base64,
            "dct_base64": dct_base64,
            "image_base64": image_base64,
            "total_blinks": total_blinks,
            "irregular_blinks": irregular_blinks,
            "full_prediction_string": full_prediction_string,
            "transcribed_text": transcribed_text,
            "similarity": similarity,
            "micro": micro,
            "gaze": gaze,
            "lip": lip,
            "mfcc1_64": mfcc1_64,
            "mfcc2_64": mfcc2_64,
            "mfcc3_64": mfcc3_64,
            "final_result": final_result,
            "result1": result1
        }




