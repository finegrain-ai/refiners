""" 
Implementation of Matte Anything from J.Yao & al (https://arxiv.org/abs/2306.04121)

Steps :

- Set foreground and background points (?);
- Give foreground caption and use grounding_Dino to detect the objects;
- Give the fg and bg points + the bounding boxes as input to SAM and get the segmentation;
- Combine all the masks;
- Generate alpha matte : 
    - Generate the trimap;
    - Using grounding Dino, detect transparent objects (given specific set of words) and annotate on the trimap (give a specific value 0.5 - transparency);
- Use ViT matte using the trimap and the original picture to get the right foreground;
- Return the extracted foreground;

"""