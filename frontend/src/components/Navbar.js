import { useState } from "react";
import { Link, useLocation } from "react-router-dom"; // Import useLocation from react-router-dom

export default function Navbar({ loginAction, text }) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation(); // Get the current path

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const isActive = (path) => location.pathname === path; // Function to check if the current path matches the link

  return (
    <nav className="fixed flex items-center justify-between bg-white border-b-2 border-black h-[8vh] lg:h-[10vh] px-2 lg:px-10 w-full font-pixeboy z-[1000] font-vt323">
      {/* Logo */}
      <div className="">
        <Link to="/">
          <img src="/logo.gif" className="h-[3vh] lg:h-[7vh]" />
        </Link>
      </div>
      {/* Desktop Menu */}
      <ul className="hidden lg:flex items-center gap-12 text-black font-bold text-3xl tracking-wider">
        <Link to="/">
          <li
            className={`hover:underline ${isActive("/") ? "text-red-500" : ""}`}
          >
            Home
          </li>
        </Link>
        <Link to="/about">
          <li
            className={`hover:underline ${isActive("/about") ? "text-red-500" : ""}`}
          >
            About
          </li>
        </Link>
        <Link to="/upload">
          <li
            className={`hover:underline ${isActive("/upload") ? "text-red-500" : ""}`}
          >
            Upload
          </li>
        </Link>
      </ul>

      {/* Mobile Menu Button */}
      <button className="lg:hidden block" onClick={toggleMenu}>
        <img
          src={isMenuOpen ? "cross-icon.svg" : "burger-menu.svg"}
          width={50}
          height={50}
          alt={isMenuOpen ? "Close Menu" : "Open Menu"}
        />
      </button>

      {/* Full-Screen Mobile Menu */}
      {isMenuOpen && (
        <div className="fixed inset-0 bg-white flex flex-col items-center justify-center text-6xl font-bold font-pixeboy z-[1001]">
          {/* Cross Icon in the Full-Screen Menu */}
          <button className="absolute top-5 right-5" onClick={toggleMenu}>
            <img
              src="cross-icon.svg"
              width={75}
              height={75}
              alt="Close Menu"
            />
          </button>

          {/* Menu Items */}
          <ul className="flex flex-col gap-8 mt-12">
            <li className="hover:underline cursor-pointer text-center">
              <Link
                to="/"
                onClick={toggleMenu}
                className={isActive("/") ? "text-red-500" : ""}
              >
                Home
              </Link>
            </li>
            <li className="hover:underline cursor-pointer text-center">
              <Link
                to="/about"
                onClick={toggleMenu}
                className={isActive("/about") ? "text-red-500" : ""}
              >
                About
              </Link>
            </li>
            <li className="hover:underline cursor-pointer text-center">
              <Link
                to="/upload"
                onClick={toggleMenu}
                className={isActive("/upload") ? "text-red-500" : ""}
              >
                Upload
              </Link>
            </li>
          </ul>
        </div>
      )}
    </nav>
  );
}
