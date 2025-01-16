import React, { useState } from 'react';
import './App.css';
import UploadForm from './components/UploadForm';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="App">
      <header className="App-header">
        <h1>PC Hardware Image Classification</h1>
        <p>Upload an image of a PC hardware component to get its classification result!</p>
      </header>

      <main>
        <UploadForm setPrediction={setPrediction} setLoading={setLoading} />
        
        {loading && <div className="spinner">Classifying...</div>}

        {prediction && !loading && (
          <div className="result">
            <h2>Prediction: {prediction.label}</h2>
            <p>Confidence: {prediction.confidence}%</p>
          </div>
        )}
      </main>

      <footer>
        <p>Made by Group 2, 02476_mlops</p>
      </footer>
    </div>
  );
}

export default App;