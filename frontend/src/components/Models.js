import React from 'react';
import "../index.css";

const About = () => {
  const info = [
    {
      head: "What are Mel-Frequency Cepstral Coefficients (MFCC), and how are they applied in audio and video processing?",
      info: (
        <>Mel-frequency cepstral coefficients (MFCC) are widely used in speech and audio processing to
        represent the short-term power spectrum of a sound signal. This is crucial for analyzing voice
        patterns in speech processing. In deepfake detection, they help identify discrepancies in audio by
        capturing subtle inconsistencies in speech characteristics in manipulated audio.</>
      ),
    },
    {
      head: "How does a digital forensics-based model detect deepfakes?",
      info: (
        <>This model performs a digital forensic analysis by examining various aspects of a video to detect
        potential manipulation. It inspects the metadata for traces of editing software or unusual codecs,
        analyzes frame consistency to identify unnatural transitions, checks facial landmarks for limited or
        abnormal movements, and scrutinizes compression artifacts for irregular patterns. These combined
        techniques allow the model to effectively assess whether a video is likely to be a deepfake.</>
      ),
    },
    {
      head: "How does LipNet work, and what role does it play in deepfake detection?",
      info: (
        <>LipNet is a deep learning model designed to read lips by analyzing video frames of a speaker's
        mouth movements. It captures the temporal dynamics of lip motion to predict spoken words without
        relying on audio input. In the context of deepfake detection, LipNet can be used to detect
        discrepancies between visual lip movements and the corresponding audio. Since deepfakes often
        have subtle misalignments between speech and lip movement, LipNet can help identify these
        inconsistencies, making it a powerful tool in detecting video-based manipulation.</>
      ),
    },
    {
      head: "What is MesoNet, and how does it contribute to identifying manipulated facial features in videos?",
      info: (
        <>MesoNet is a deep learning architecture specifically designed for deepfake detection, focusing on
        identifying manipulated facial features in videos. It analyzes mesoscopic properties—mid-level
        patterns in facial images that are altered during the creation of deepfakes. By detecting subtle
        artifacts and distortions introduced by deepfake generation methods, MesoNet can differentiate
        between real and fake faces.</>
      ),
    },
    {
      head: "How does gaze tracking improve deepfake detection?",
      info: (
        <>Gaze tracking is a technique that monitors eye movements to detect irregularities, such as abnormal
        blinking intervals or unusual blink counts, which are often signs of deepfake manipulations. By
        calculating the eye aspect ratio (EAR), these irregularities can be quantified to identify mismatches
        between expected and observed blink patterns, thus helping in detecting deepfakes.</>
      ),
    },
  ];

  const getItems = () => {
    return info;
  };

  return (
    <div className="flex flex-col w-full h-full about overflow-hidden pt-[10vh] flex flex-col items-center justify-center overflow-hidden">
      <div className="w-[90vw] lg:w-[95vw] h-auto border-black my-12 border-4 shadow-[-5px_5px_0_0_#000000] lg:shadow-[-10px_10px_0_0_#000000] flex flex-col items-center">
            <div className="bg-white h-[2vh] lg:h-[8vh] w-full flex items-center border-black border-b-2 pl-[0.5vw]">
                <img src="/signal.svg" alt="Traffic Signal" className="h-[1vh] lg:h-[5vh]" />
            </div>
      <div className="flex flex-col justify-center items-center h-1/5 font-protest border-b-4 lg:border-b-8 border-white w-full">
        <h1 className="text-white text-[3vh] lg:text-[6vh] font-vt323">About the Models</h1>
      </div>

      <div className="flex flex-col items-center  w-full ">
        
        {getItems().map((item, index) => (
          <div
            key={index}
            className={`flex font-vt323 border-b-2 lg:border-b-4 border-white py-6 w-full  ${
              index % 2 === 0 ? "flex-row-reverse" : "flex-row"
            }`}
          >
            <div className="flex-1">
              <h2 className="text-[3vh] lg:text-[5vh] font-bold px-4 text-cyan-300">{item.head}</h2>
              <p className="text-[2.5vh] lg:text-[4vh] mt-2 text-yellow-400 font-semibold px-8">{item.info}</p>
            </div>
            
          </div>
        ))}
      </div>
    </div>
    </div>
  );
};

export default About;
