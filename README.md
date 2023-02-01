# Vision Langauge Models simplified (torch implementations)
### Simple implementation of Vision Language models in torch

This repo aims at demonstrating implementation of prominent models in torch in vision-langauge models.

* Current implementations:
    *   CLIP model :
        * Original paper: https://arxiv.org/abs/2103.00020
        * Official implementation: https://github.com/openai/CLIP
        * Source motivated and borrowed from : https://github.com/moein-shariatnia/OpenAI-CLIP
        * Training model is available in form of modules and scripts.
        * Zero-shot classification inference mode is implemented in form of notebook.
    * LIT model :
        * Original paper: https://arxiv.org/abs/2111.07991
        * Official implementation: https://github.com/google-research/vision_transformer/blob/main/model_cards/lit.md
        * In nutshell, LiT differs from the original CLIP, in a sense that the LiT freezes the image tower and trains and tunes the text model.
    * CoCo model :
        * Original paper: https://arxiv.org/pdf/2205.01917v2.pdf
        * Upcoming.
