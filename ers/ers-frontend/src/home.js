import React from 'react';
import { useNavigate } from 'react-router-dom';
import './home.css';
import logo from './logo.svg'; // Make sure this path is correct

const Home = () => {
  const navigate = useNavigate();

  const handleGetOutput = () => {
    navigate('/output');
  };

  return (
    <div className="container">
      <header className="header">
        <div className="logo-container">
          <img src={logo} className="logo" alt="logo" />
          <div className="company-name">
            <span className="font-weight-1">E</span>
            <span className="font-weight-2">V</span>
            <span className="font-weight-3">E</span>
            <span className="font-weight-4">R</span>
            <span className="font-weight-5">S</span>
            <span className="font-weight-6">A</span>
            <span className="font-weight-7">N</span>
            <span className="font-weight-8">A</span>
          </div>
        </div>
        <button className="sign-up">SIGN UP</button>
      </header>
      <main className="main-content">
        <div className="content-wrapper">
          <div className="left-side">
            <h3>Enter Text Here</h3>
            <textarea 
              className="text-input" 
              placeholder="Enter your text here..."
            />
          </div>
          <div className="right-side">
            <div className="model-list">
              <h3>MODEL LIST</h3>
              <div className="model-options">
                {/* Add model options here */}
              </div>
            </div>
            <button className="get-output" onClick={handleGetOutput}>Get Output</button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Home;