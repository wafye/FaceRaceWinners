# FaceRaceWinners
Winners

Both of the basic python scripts that I put up (haar_cascades and mixture of gaussians)
are raw basic code as proof of concept. Needs to be altered before final implementation.

For Mixture of Gaussians: The current implementation does not segment only the head, it segments
the entire object that is moving in the frame which typically being the entire person.

For Haar_Cascades: We can set a minimum face threshold size (60x60px) and I would suggest that
when we implement it, we do a sequential frame check to handle wrong face guestimates.

For face_recognizer: You must download the yalefaces database at the bottom of this link:
http://hanzratech.in/2015/02/03/face-recognition-using-opencv.html#comment-2130189209
(The python program should be in the same directory as the yalefaces folder)
