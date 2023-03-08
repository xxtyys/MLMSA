# Multimodal Sentiment Analysis Using Multiple Labels from Different Modalities


## Data and Environment
- Available at https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
- Download and extract images and texts into separate folders in `data/mvsa_single/` and `data/mvsa_multiple/`
- Create an environment using environment.yml with conda
## Splits
- 10 Fold Train/Val/Test splits provided in data/ for MVSA-single and MVSA-multiple.
- valid_pairlist.txt format is `file_id (filename), multimodal label, text label, image label`
- 0 (Neutral), 1 (Positive), 2 (Negative)
- Split file rows point to the line number in valid_pairlist.txt (0-indexed)
- `multimodal label` is used for training and evaluating all the models.

## Train and evaluate models
`bash shells/run_singlesplit.sh`