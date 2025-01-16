import React, { useState } from 'react';
import axios from 'axios';

const UploadForm = ({ setPrediction, setLoading }) => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setImage(file);
        setImagePreview(URL.createObjectURL(file)); // Preview the image
        setErrorMessage('');
      } else {
        setErrorMessage('Please upload a valid image file.');
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setErrorMessage('Please select an image!');
      return;
    }

    setLoading(true);
    setErrorMessage(''); // Clear previous error

    // Prepare the form data to send to the backend API
    const formData = new FormData();
    formData.append('file', image);

    try {
      // Send the image to the backend for classification
      const response = await axios.post(process.env.REACT_APP_API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Update the prediction result
      setPrediction(response.data);
    } catch (error) {
      console.error('Error uploading the image:', error);
      setErrorMessage('Error uploading the image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-form">
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleImageChange} />
        <button type="submit">Upload and Classify</button>
      </form>

      {errorMessage && <div className="error-message">{errorMessage}</div>}

      {imagePreview && (
        <div className="image-preview">
          <h3>Image Preview:</h3>
          <img src={imagePreview} alt="Preview" className="preview-image" />
        </div>
      )}
    </div>
  );
};

export default UploadForm;