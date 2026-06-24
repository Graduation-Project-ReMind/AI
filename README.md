# 👁️ ReMind - AI Eye-Tracking & Attention Management System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/OpenCV-Latest-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Latest-red.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/WebSockets-Active-orange.svg" alt="WebSockets">
</p>

---

## 📝 Overview

The **ReMind AI Engine** is the core cognitive tracking component of the project. It utilizes advanced **Computer Vision** and **Deep Learning frameworks** to perform real-time eye-tracking and gaze estimation via the user's mobile camera. 

The primary objective is to analyze cognitive focus, detect attention shifts, and provide metric reports to assist in evaluating individuals with ADHD or attention-deficit challenges.

---

## 🧠 Core Features

* **Real-time Gaze Estimation:** Tracks eye movement and maps iris/pupil positioning.
* **Attention Scoring Algorithm:** Computes focal coordinates and calculates a dynamic attention percentage based on fixation periods.
* **Blink Detection & Fatigue Analysis:** Monitors blink rates to distinguish between natural fatigue and distraction.
* **Low-Latency Streaming:** Powered by high-performance WebSocket communication to deliver seamless frame-by-frame analysis between the Flutter app and the AI server.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.9+
* **Face Mesh & Landmark Detection:** `MediaPipe` (Facial landmarker with 468+ points)
* **Image Processing:** `OpenCV`
* **Real-Time Communication:** `WebSockets` / `asyncio`
* **Data Crunching:** `NumPy` & `SciPy`

---

## 🏗️ System Architecture
