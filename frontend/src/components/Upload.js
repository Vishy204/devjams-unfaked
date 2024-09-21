import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function UploadPage() {
    const [videoFile, setVideoFile] = useState(null);
    const [audioFile, setAudioFile] = useState(null);
    const [isLoading, setIsLoading] = useState(false); // Loader state
    const navigate = useNavigate();

    // Handle video file input change
    const handleVideoFileChange = (e) => {
        setVideoFile(e.target.files[0]);
    };

    // Handle audio file input change
    const handleAudioFileChange = (e) => {
        setAudioFile(e.target.files[0]);
    };

    // Handle the upload of both audio and video
    const handleUpload = async () => {
        if (!videoFile && !audioFile) {
            alert("Please select a video or audio file.");
            return;
        }

        const formData = new FormData();
        if (videoFile) {
            formData.append('video', videoFile); // Add video file if selected
        }
        if (audioFile) {
            formData.append('audio', audioFile); // Add audio file if selected
        }

        try {
            setIsLoading(true); // Start loading
            const response = await axios.post('http://127.0.0.1:8000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            const { result, random_array,  prediction, frame_base64, dct_base64, image_base64, total_blinks, irregular_blinks, full_prediction_string, transcribed_text, similarity, micro,  gaze, lip, mfcc1_64, mfcc2_64, mfcc3_64, final_result,result1 } = response.data;

            // Navigate to ResultPage with the result, randomArray, metadata, and encoded images
            navigate('/result', {
                state: {
                    result,
                    randomArray: random_array || [],
                    prediction,
                    frame_base64,  // Add frame_base64 image
                    dct_base64,    // Add dct_base64 image
                    image_base64,
                    total_blinks,
                    irregular_blinks,
                    full_prediction_string,
                    transcribed_text,
                    similarity,
                    micro,  gaze, lip, mfcc1_64, mfcc2_64, mfcc3_64, final_result ,result1// Add transcribed_text
                }
            });
        } catch (error) {
            console.error('Error uploading the files:', error);
        } finally {
            setIsLoading(false); // Stop loading
        }
    };

    return (
        <div className="text-white pt-[10vh] about w-full h-auto flex flex-col items-center justify-center overflow-hidden">
            
            <div className="w-[90vw] lg:w-[95vw] h-auto border-black my-12 border-4 shadow-[-5px_5px_0_0_#000000] lg:shadow-[-10px_10px_0_0_#000000] flex flex-col items-center">
            <div className="bg-white h-[2vh] lg:h-[8vh] w-full flex items-center border-black border-b-2 pl-[0.5vw]">
                <img src="/signal.svg" alt="Traffic Signal" className="h-[1vh] lg:h-[5vh]" />
            </div>
            <div className="flex flex-col text-white w-full">
    <div className="text-center w-full">
        <p className="text-[3vh] lg:text-[6vh] font-bold my-6 font-vt323">
            Welcome to UnFakeable – Your Reliable Deepfake Detection Tool
        </p>
    </div>
    <div className="lg:flex justify-center border-b-4 lg:border-b-8 border-white py-8 w-full">
    <p className="text-[2.5vh] lg:text-[4vh] mb-6 font-vt323 w-full lg:w-[50vw] px-4 text-center lg:text-left">
        In an era where media manipulation is becoming increasingly sophisticated, detecting the
        authenticity of video and audio content is more important than ever. At UnFakeable, we
        leverage cutting-edge AI technology to help you identify whether a file is genuine or
        manipulated, specifically targeting deepfake content.
    </p>
    <img src="/upload.jpeg" className="w-[70vw] lg:w-[30vw] h-[25vh] lg:h-[40vh] mx-auto lg:mx-0 lg:inline" />
</div>

    <h2 className="text-[3vh] lg:text-[6vh] font-semibold my-8 font-vt323 text-center">
        How It Works:
    </h2>
    <ul className="custom-square-list pl-8 lg:pl-32">
        <li>
            <div className="text-[2vh] lg:text-[4vh] mb-4 text-left font-vt323">
                <strong>Upload Your Media:</strong> You can upload either a video (in .mp4 format) or an
                audio file (in .wav format).
            </div>
        </li>
        <li>
        <div className="text-[2vh] lg:text-[4vh] mb-4 text-left font-vt323">
                <strong>Real-Time Analysis:</strong> Once your file is uploaded, our system will analyze it
                using advanced deep learning models trained to detect inconsistencies typical of deepfakes.
            </div>
        </li>
        <li>
        <div className="text-[2vh] lg:text-[4vh] mb-4 text-left font-vt323">
                <strong>Instant Results:</strong> After processing, you will receive a clear report
                indicating whether the content is real or a deepfake, along with detailed reasons behind the
                classification.
            </div>
        </li>
    </ul>
</div>



            <div className="pb-12 flex flex-col items-center justify-center text-center font-vt323">
                <h1 className="text-white text-[2.5vh] lg:text-[3vh] underline">Upload Audio (.wav) or Video for Analysis</h1>

                {/* Video Upload */}
                <div className="my-4 lg:my-6 text-white text-[2vh] lg:text-[3vh]">
                    <label>Upload Video:</label>
                    <input type="file" onChange={handleVideoFileChange} accept="video/*" className="" />
                </div>

                {/* Audio Upload */}
                <div className="my-4 lg:my-6 text-white text-[2vh] lg:text-[3vh]">
                    <label>Upload Audio (.wav):</label>
                    <input type="file" onChange={handleAudioFileChange} accept="audio/wav" />
                </div>

                <button onClick={handleUpload} className="p-2 lg:p-4 bg-red-500 hover:bg-yellow-500 border-[#525252] hover:border-black border-4 shadow-[-10px_10px_0_0_#525252] hover:shadow-[-10px_10px_0_0_#000000]">
                    Upload and Process
                </button>

                {isLoading && (
                    <div class="loader mt-8"></div>
                )}
            </div>
            </div>
        </div>
    );
}

export default UploadPage;
