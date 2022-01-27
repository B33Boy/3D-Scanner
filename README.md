# 3D-Scanner
<!---

To collect images for camera calibration, run the following code in the `3D-Scanner/` directory:

`python scripts/collect_cali.py`


To calibrate the camera, run the following code in `3D-Scanner/` directory:

`python scripts/calibrate.py`

-->


<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url] 
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
<!--
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
-->
  <h3 align="center">Capstone Project: Handheld Laser Scanner</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!--- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

Some stuff about the project

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

You must have linux installed with OpenCV 4.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Change to rot directory
   ```sh
   cd 3D-Scanner
   ```
 

Run `python scripts/collect_cali.py` to take between 10 to 20 images of the 7x9 checkerboard. If another checkerboard is used, change the `CHECKERBOARD` variable in the script.

- press `q` to quit the application
- press `c` to take an image
- press `z` to pop the last image

Next run `python scripts/calibrate.py` to run the calibration algorithm. The camera matrix, disortion coefficients, rotation, and translation vectors will be exported to `res/cal_out/`.


## Usage

### Generating requirements.txt
run the command `./gen_pipreq.sh` from the root of the project to update the requirements.txt



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Abhi Patel - abhi.patel@ontariotechu.net

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Acknowledge Dr. Barari & Cody Berry here

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/B33Boy/3D-Scanner.svg?style=for-the-badge
[contributors-url]: https://github.com/B33Boy/3D-Scanner/graphs/contributors
[license-shield]: https://img.shields.io/github/license/B33Boy/3D-Scanner.svg?style=for-the-badge
[license-url]: https://github.com/B33Boy/3D-Scanner/blob/main/LICENSE.txt

