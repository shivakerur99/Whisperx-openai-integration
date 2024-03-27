# Alindor-app-

# Frontend

first create your virtual environment in your machine using command [vitualenv "name of your env"]
then activate your environment using command ["nameofyourenvironment"\Scripts\activate] for deactivation of your environment run command ["nameofyourenvironment"\Scripts\deactivate]

first for frontend go to alindor-app folder
cmd [cd alindor-app]
run command [npm install]
next step
run command [npm start]

go to Api.js and mention your backend server running on which port and server it is running on

# Backend

first create your virtual environment in your machine using command [vitualenv "name of your env"]
then activate your environment using command ["nameofyourenvironment"\Scripts\activate] for deactivation of your environment run command ["nameofyourenvironment"\Scripts\deactivate]

first for Backend go to FastApi folder

cmd [ cd FastApi]

and install all the necessary packages using command

pip install -r requirements.txt 

then run command [uvicorn main:app --reload]


# steps to run the app for [deployed]

3.. Grandmaster Mode

Github repo= https://github.com/shivakerur99/alindor-grandmaster-mode

hugging face repo = https://huggingface.co/spaces/shivakerur99/alindor_grandmaster/tree/main

Frontend deployed app link = https://alindor-grandmaster-mode.onrender.com/

Backend deployed app link = https://shivakerur99-alindor-grandmaster.hf.space

video of working app in local machine and vc also= https://drive.google.com/file/d/1hf2hmfT5BosB_cc-7DAn0Sk6RHcadYX6/view?usp=sharing

I have added a docker file in the hugging face repo I haven't added a docker file in the GitHub repo and we should have ffmpeg.exe in the project path before running the app i have added a readme file on how to run the app. Kindly check out the GitHub repo for this 

in grandmaster mode I faced many challenges just to load the model due to less time I couldn't reduce process time for transcription further down, process time takes longer than deepgram API still whisperx is better than deepgram API in terms of speaker diarization.

(as the assignment said to host on the Google Cloud platform but I used hugging Face is simple it provides me 16 gb CPU ram vc instance which is free to use. I couldn't deploy this on render because it provides 512 MB RAM which is too low for Whisperx open source library to load)

disadvantage = overall process time is very slow   

we can use openAI subscription version to reduce process time 

note:= For 3 sentences to process it takes 1 min like for 30 sentences it takes 10 mins   

click on + button to upload file and then (important) click on upload txt 

click on get analysis button wait till it process (if we use 3 lines txt file then 2 mins to process as i am using free version so i added time sleep in code to process it, if we used subscribed version we can increase request per minute)

