import os
import json
import joblib
import pandas as pd
from collections import defaultdict
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error



class LLMEnhancedRecSys:
    def __init__(self):
        self.is_trained = False
        self.weights_path = "./registry/llm_enhanced_recsys.joblib"

        self.user_encoder = OneHotEncoder(handle_unknown="ignore")
        self.item_encoder = OneHotEncoder(handle_unknown="ignore")
        self.text_vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2
        )
        self.model = Ridge(alpha=1.0)

    def _load_dataset(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            return pd.read_csv(path)

        if ext in [".json", ".jsonl"]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                raise ValueError(f"Empty dataset: {path}")

            # 1. Try normal JSON array / single object
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    data = [data]
                return pd.DataFrame(data)
            except json.JSONDecodeError:
                pass

            # 2. Try true JSONL: one full JSON object per line
            rows = []
            jsonl_ok = True
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    jsonl_ok = False
                    break

            if jsonl_ok and rows:
                return pd.DataFrame(rows)

            # 3. Try concatenated pretty-printed JSON objects
            decoder = json.JSONDecoder()
            rows = []
            idx = 0
            length = len(content)

            while idx < length:
                while idx < length and content[idx].isspace():
                    idx += 1
                if idx >= length:
                    break

                obj, end = decoder.raw_decode(content, idx)
                rows.append(obj)
                idx = end

            if rows:
                return pd.DataFrame(rows)

            raise ValueError(f"Could not parse dataset file: {path}")

        raise ValueError("Supported file types: .csv, .json, .jsonl")

    def _round_to_half(self, value: float) -> float:
        value = max(1.0, min(5.0, value))
        return round(value * 2) / 2

    def _build_profiles_from_train(self, train_df: pd.DataFrame):
        user_reviews = defaultdict(list)
        item_reviews = defaultdict(list)

        for _, row in train_df.iterrows():
            user_reviews[row["user_id"]].append(str(row["text"]))
            item_reviews[row["item_id"]].append(str(row["text"]))

        user_rating_map = train_df.groupby("user_id")["stars"].mean().to_dict()
        item_rating_map = train_df.groupby("item_id")["stars"].mean().to_dict()

        user_profiles = {}
        item_profiles = {}

        for uid, reviews in user_reviews.items():
            joined = " ".join(reviews[:50])
            avg = user_rating_map.get(uid, 3.5)
            user_profiles[uid] = {
                "user_profile": f"user_avg_stars={avg:.2f} past_reviews={joined}"
            }

        for iid, reviews in item_reviews.items():
            joined = " ".join(reviews[:50])
            avg = item_rating_map.get(iid, 3.5)
            item_profiles[iid] = {
                "item_profile": f"item_avg_stars={avg:.2f} past_reviews={joined}"
            }

        return user_profiles, item_profiles

    def _attach_profiles(self, df: pd.DataFrame, user_profiles: dict, item_profiles: dict) -> pd.DataFrame:
        df = df.copy()

        df["user_profile"] = df["user_id"].map(
            lambda x: user_profiles.get(x, {"user_profile": "user_avg_stars=3.5"})["user_profile"]
        )
        df["item_profile"] = df["item_id"].map(
            lambda x: item_profiles.get(x, {"item_profile": "item_avg_stars=3.5"})["item_profile"]
        )

        df["combined_text"] = df["user_profile"].fillna("") + " [SEP] " + df["item_profile"].fillna("")
        return df

    def _build_features(self, df: pd.DataFrame, fit: bool = False):
        user_col = df[["user_id"]]
        item_col = df[["item_id"]]
        text_col = df["combined_text"].fillna("")

        if fit:
            X_user = self.user_encoder.fit_transform(user_col)
            X_item = self.item_encoder.fit_transform(item_col)
            X_text = self.text_vectorizer.fit_transform(text_col)
        else:
            X_user = self.user_encoder.transform(user_col)
            X_item = self.item_encoder.transform(item_col)
            X_text = self.text_vectorizer.transform(text_col)

        return hstack([X_user, X_item, X_text])

    def train_with_dataset(self, dataset_path: str):
        print("\n" + "=" * 60)
        print("Starting offline training...")
        print("=" * 60)

        # ==================================================
        # STEP 1. Load data + split into train / val / test
        # ==================================================
        df = self._load_dataset(dataset_path)

        required_cols = {"user_id", "item_id", "stars", "text"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.dropna(subset=["user_id", "item_id", "stars", "text"]).copy()
        df["stars"] = df["stars"].astype(float)
        df["text"] = df["text"].astype(str)

        # 80% train, 10% val, 10% test
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        print(f"Train size: {len(train_df)}")
        print(f"Val size:   {len(val_df)}")
        print(f"Test size:  {len(test_df)}")

        # Build profiles ONLY from train set to avoid leakage
        user_profiles, item_profiles = self._build_profiles_from_train(train_df)

        train_df = self._attach_profiles(train_df, user_profiles, item_profiles)
        val_df = self._attach_profiles(val_df, user_profiles, item_profiles)
        test_df = self._attach_profiles(test_df, user_profiles, item_profiles)

        # ==================================================
        # STEP 2. Train on train set, evaluate on val set
        # ==================================================
        X_train = self._build_features(train_df, fit=True)
        y_train = train_df["stars"].values

        X_val = self._build_features(val_df, fit=False)
        y_val = val_df["stars"].values

        self.model.fit(X_train, y_train)

        val_pred = self.model.predict(X_val)

        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = mean_squared_error(y_val, val_pred) ** 0.5

        rounded_val_pred = [self._round_to_half(x) for x in val_pred]
        val_rounded_accuracy = sum(
            p == y for p, y in zip(rounded_val_pred, y_val)
        ) / len(y_val)
        val_within_half = sum(
            abs(p - y) <= 0.5 for p, y in zip(rounded_val_pred, y_val)
        ) / len(y_val)

        print("\n===== VALIDATION RESULTS =====")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Validation Rounded Accuracy: {val_rounded_accuracy:.4f}")
        print(f"Validation Within 0.5 star Accuracy: {val_within_half:.4f}")

        # Save trained model
        os.makedirs("./registry", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "user_encoder": self.user_encoder,
                "item_encoder": self.item_encoder,
                "text_vectorizer": self.text_vectorizer,
                "user_profiles": user_profiles,
                "item_profiles": item_profiles,
            },
            self.weights_path
        )

        # ==================================================
        # STEP 3. Final evaluation on test set
        # ==================================================
        X_test = self._build_features(test_df, fit=False)
        y_test = test_df["stars"].values

        test_pred = self.model.predict(X_test)

        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = mean_squared_error(y_test, test_pred) ** 0.5

        rounded_test_pred = [self._round_to_half(x) for x in test_pred]
        test_rounded_accuracy = sum(
            p == y for p, y in zip(rounded_test_pred, y_test)
        ) / len(y_test)
        test_within_half = sum(
            abs(p - y) <= 0.5 for p, y in zip(rounded_test_pred, y_test)
        ) / len(y_test)

        print("\n===== FINAL TEST RESULTS =====")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test Rounded Accuracy: {test_rounded_accuracy:.4f}")
        print(f"Test Within 0.5 star Accuracy: {test_within_half:.4f}")

        self.is_trained = True
        print(f"\nModel saved to: {self.weights_path}")

    def load(self):
        bundle = joblib.load(self.weights_path)
        self.model = bundle["model"]
        self.user_encoder = bundle["user_encoder"]
        self.item_encoder = bundle["item_encoder"]
        self.text_vectorizer = bundle["text_vectorizer"]
        self.user_profiles = bundle["user_profiles"]
        self.item_profiles = bundle["item_profiles"]
        self.is_trained = True

    def predict(self, user_id: str, item_id: str) -> float:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call train_with_dataset() or load().")

        row = pd.DataFrame([{
            "user_id": user_id,
            "item_id": item_id,
            "user_profile": self.user_profiles.get(
                user_id, {"user_profile": "user_avg_stars=3.5"}
            )["user_profile"],
            "item_profile": self.item_profiles.get(
                item_id, {"item_profile": "item_avg_stars=3.5"}
            )["item_profile"],
        }])

        row["combined_text"] = row["user_profile"] + " [SEP] " + row["item_profile"]

        X = self._build_features(row, fit=False)
        pred = float(self.model.predict(X)[0])
        return self._round_to_half(pred)