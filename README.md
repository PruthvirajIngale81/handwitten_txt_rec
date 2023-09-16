# Handwritten Text Recognition

This repository contains code and resources for a Handwritten Text Recognition (HTR) system. The goal of this project is to recognize and transcribe handwritten text from images into machine-readable text.

## Installation

 * Clone the repository:

   ```
     git clone https://github.com/PruthvirajIngale81/handwitten_txt_rec.git
   ```

   ```
     cd handwitten_txt_rec
   ```
# Run locally (manual)   

1. install required dependencies:


```
   pip3 install -r requirements.txt
```
 2. run this cmd: make sure you have latest python version


```
python3 app.py
```
 3. Hurry access using following URL:

`https://0.0.0.0:5000`

# Docker way (ubantu):
1. install docker:

   * update your machine:


     ```
       sudo apt-get update
     ```

   * install docker into machine:


     ```
       sudo apt-get install docker.io
     ```
2. build docker image:


```
sudo docker build -t handwrittentxt:v1 .
```
3. deploy you app into container and run application using:


```
sudo docker run -d -p 5000:5000 handwrittentxt:v1
```
Hurry access your application using :


~ For local machine: `https://0.0.0.0:5000`


~ For remote host: `https://<ip_address>:5000`

### Happy shipping !