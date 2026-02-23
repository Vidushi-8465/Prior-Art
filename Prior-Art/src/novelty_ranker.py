from sentence_transformers import SentenceTransformer, util
import pandas as pd

class NoveltyRanker:
    def __init__(self, csv_path="../data/output/batch_patent_analysis.csv"):
        self.df = pd.read_csv(csv_path)
        print("\nInitializing Novelty Ranker...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.df = pd.read_csv(csv_path)
        self.texts = self.df['abstract'].fillna("").tolist()
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
        print(f"Loaded {len(self.texts)} existing patents for comparison.\n")

    def rank_novelty(self, user_idea: str, top_k: int = 5):
        """Rank existing patents against the user's idea."""
        idea_embedding = self.model.encode(user_idea, convert_to_tensor=True)
        cosine_scores = util.cos_sim(idea_embedding, self.embeddings)[0]
        results = []
        for idx in cosine_scores.argsort(descending=True)[:top_k]:
            idx = int(idx.item())  # ✅ minimal fix – converts tensor to int safely
            results.append({
                "rank": len(results) + 1,
                "title": self.df.iloc[idx]["title"],
                "similarity": float(cosine_scores[idx]),
                "novelty_score": round((1 - float(cosine_scores[idx])) * 100, 2)
            })
        return results