# Task 2 Findings: Painting Similarity (NGA Open Data)

## 1. Objective
Build a painting similarity system that retrieves visually similar artworks, with a focus on portraits, using the National Gallery of Art open dataset.

## 2. Data Source
- Dataset root: ../opendata-main/data
- Main tables used:
  - published_images.csv
  - objects.csv
  - objects_constituents.csv
  - constituents.csv
  - objects_terms.csv

## 3. Data Construction
The retrieval dataset is built by:
1. Keeping primary image views from published_images.
2. Linking images to object metadata.
3. Linking objects to artists using roletype = artist.
4. Creating portrait flags using term/title text containing portrait.
5. Keeping artists with at least 3 images for stable retrieval labels.
6. Sampling up to 800 images for efficient experimentation.

## 4. Model and Similarity Strategy
Approach used:
1. Download and cache thumbnail images from iiifthumburl.
2. Extract visual embeddings with a pretrained ResNet50 encoder.
3. L2-normalize embeddings.
4. Compute cosine similarity for nearest-neighbor retrieval.

Why this is appropriate:
- No direct pairwise similarity labels are provided in NGA tables.
- Pretrained visual embeddings are a strong baseline for semantic retrieval.
- Artist identity is a practical proxy relevance label for benchmarking.

## 5. Evaluation Metrics
Metrics used for ranking quality:
1. Recall@K (K = 1, 5, 10): whether at least one relevant image appears in top-K.
2. mAP: rewards ranking relevant items early across the list.
3. nDCG: position-sensitive ranking quality versus ideal ordering.

Relevance definition for evaluation:
- Relevant if query and candidate share the same artist_id.

Two evaluation settings:
1. All paintings query subset.
2. Portrait-only query subset.

## 6. Results from Latest Run
All paintings:
- Recall@1: 0.1050
- Recall@5: 0.1750
- Recall@10: 0.2200
- mAP: 0.0934
- nDCG: 0.1781

Portrait subset:
- Recall@1: 0.2097
- Recall@5: 0.2903
- Recall@10: 0.3710
- mAP: 0.2015
- nDCG: 0.2927

## 7. Interpretation
1. Portrait retrieval is significantly stronger than global retrieval in this setup.
2. This suggests embeddings capture shared face/pose/composition cues reasonably well.
3. Some visually similar results are from different artists, which is expected and indicates semantic similarity beyond artist identity.

## 8. Limitations
1. Artist-based relevance can under-score true visual similarity across artists.
2. Thumbnail resolution is limited and may lose fine stylistic details.
3. This is a zero-shot baseline (no metric-learning fine-tuning).

## 9. Recommended Next Improvements
1. Use CLIP or DINOv2 embeddings for stronger semantic alignment.
2. Fine-tune with triplet or contrastive loss using artist/style positives.
3. Add re-ranking with portrait-specific descriptors for face and pose emphasis.
4. Expand evaluation with manual relevance judgments for a small curated query set.

## 10. Reproducibility Notes
- Notebook: task2_similarity/task2.ipynb
- Sampling seed: 42
- Max sampled images: 800
- Min images per artist: 3
