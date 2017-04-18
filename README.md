# FaceRaceWinners
Winners

Both of the basic python scripts that I put up (haar_cascades and mixture of gaussians)
are raw basic code as proof of concept. Needs to be altered before final implementation.

For Mixture of Gaussians: The current implementation does not segment only the head, it segments
the entire object that is moving in the frame which typically being the entire person.

For Haar_Cascades: We can set a minimum face threshold size (60x60px) and I would suggest that
when we implement it, we do a sequential frame check to handle wrong face guestimates.
