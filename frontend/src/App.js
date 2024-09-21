import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import './App.css';
import UploadPage from './components/Upload';
import Models from "./components/Models"
import Navbar from "./components/Navbar";
import Home from "./components/Home"
import ResultPage from "./components/Result";

function App() {
  return (
    <Router>
    <Routes>
    <Route path="/" element={<div><Navbar /><Home /></div>} />
    <Route path="/upload" element={<div><Navbar /><UploadPage /></div>} />
    <Route path="/about" element={<div><Navbar /><Models /></div>} />
    <Route path="/result" element={<div><Navbar /><ResultPage /></div>} />

    </Routes>
    </Router>
  );
  
}

export default App;
