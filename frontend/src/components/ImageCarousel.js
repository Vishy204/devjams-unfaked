import React, { useState, useEffect } from 'react';
import { Carousel } from 'react-responsive-carousel';
import 'react-responsive-carousel/lib/styles/carousel.min.css';

const ImageCarousel = () => {
  const [loading, setLoading] = useState(true);
  const [loadedImages, setLoadedImages] = useState(0);
  const totalImages = 3;

  const handleImageLoad = () => {
    setLoadedImages(prev => prev + 1);  // Correct use of 'prev'
    console.log("img loaded");
  };

  const checkIfImageLoaded = (imgElement) => {
    if (imgElement.complete && imgElement.naturalHeight !== 0) {
      handleImageLoad();
      console.log("handle img load");
    }
  };

  useEffect(() => {
    if (loadedImages >= totalImages) {
      setLoading(false);
    }
    console.log(`Total loaded images: ${loadedImages}`); // Log updated state here
  }, [loadedImages]);

  useEffect(() => {
    // Check if images are already loaded from cache when component mounts
    const images = document.querySelectorAll('img');
    images.forEach(img => checkIfImageLoaded(img));
  }, []);

  return (
    <div>
      {loading && (
        <div className="flex items-center justify-center h-[50vh] lg:h-[85vh]">
          <div class="loader"></div>
        </div>
      )}
      <div style={{ display: loading ? 'none' : 'block' }}>
        <Carousel showThumbs={false} autoPlay infiniteLoop>
          <div className="h-[50vh] lg:h-[85vh]">
            <img src="/img-car-1-com.jpeg" alt="Slide 1" className="h-full" onLoad={handleImageLoad} />
          </div>
          <div className="h-[50vh] lg:h-[85vh]">
            <img src="/img-car-2-com.jpeg" alt="Slide 2" className="h-full" onLoad={handleImageLoad} />
          </div>
          <div className="h-[50vh] lg:h-[85vh]">
            <img src="/img-car-3-com.jpeg" alt="Slide 3" className="h-full" onLoad={handleImageLoad} />
          </div>
        </Carousel>
      </div>
    </div>
  );
};

export default ImageCarousel;
