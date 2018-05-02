# Voice-recognition
Voice recognition using FFT and Google Speech API ready to be adapted for any system that can take a high/low input or to be coded on to pi outputs. Note google api method only supports Pi3 currently. 

Depenency libraries are numpy,scipy,PeakUtils,PyAudio,google-cloud-speech.

Built for python3 can be easily adapted for python2. 

Requires google console project for authorizing api calls. Found at https://cloud.google.com/speech-to-text/docs/quickstart-client-libraries.

Credit to Cryo for the best method to record trimmed files to wav format found here https://stackoverflow.com/questions/892199/detect-record-audio-in-python and google for the methods to send a wav file to the API to be converted to text.

# Files

Registration.py - Set up stage builds file of stored audio recordings of user selected passwords that are stored in Prompts.txt. Can be used for multiple users as long as selected name is not already taken.

Computer_Demo.py - Waits for keyboard interrupt then asks for username. (Currently coded to my name to save repeating in testing.) Comment out line 253 and uncomment the 3 lines above to allow for multiple users or change name on line 253 for single user set up. Code will then ask for password. Say one of the words you have registered. This will then be compared and will either enter a 'open' state or remain 'closed' depending on the acceptance limits and how many of the first major peaks are the same. 

2 wav gen.py - Creates 2 wav files one after the other for using with test2.py to refine variables for your specific operational enviroment.

test2.py - Analyses the wav files created by 2 wav gen.py and shows the peak locations, number of matches and number of peaks to compare against. Also features ploting and text file outputs of data for analysis. Using the plots you can identify the peaks on the frequency diagram and adjust the variables. 

Pi_Demo.py - Optimised varables without plotting for my enviroment. Otherwise the same as Computer demo

 # Variables to be adjusted:

Threshold - Magnitiude that if below will not trigger microphone and once above when the value falls below Threshold will cut off again.(Pi_demo global var(line 22))

fdrift	- Wiggle room on difference in peak location (Pi_demo method open_sesame(line 117))

acceptance	- ratio to amount of peaks correct to total required for a pass condition. (Pi_demo method open_sesame(line 127))

buffer 	- Recording technique to help speech api distiguish words adds silence on either side of the recording. Used with lowering Threshold can reduce clipping on short phrases. (Pi_demo method record(line 223))

max peaks	- max number of peaks analysed to many will make it harder to get repeatable results. Too few will make the system easier to open. (Pi_demo method find_peaks(line 101))

Peak Threshold 	- put in a number of how significant a peak must be compared to the maximum. Ie 0.99 only max will show. Lower value for more peaks adjust by referencing the plots generated. (Pi_demo method find_peaks(line 78))

Peak Min dist		- Minumum distance between peaks to define local maximum. (Pi_demo method find_peaks(line 78))
