import React from 'react';
import './home.css';
import logo from './logo.svg'; // Assume you have this image in your project

const Output = () => {
  return (
    <div className="container">
      <div className="header">
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
      </div>
      <div className="content">
        <div className="output-box">
        <h3>Output</h3>
            <textarea 
              className="text-output" 
              placeholder="Output displays here"
            />
        </div>
        <div className="entity-list">
          <h2>ENTITY LIST</h2>
          <ul>
            {['red', 'olive', 'green', 'teal', 'blue', 'purple'].map((color, index) => (
              <li key={index}>
                <span className={`dot ${color}`}></span>
                <span className="line"></span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Output;