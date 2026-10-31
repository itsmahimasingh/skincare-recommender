# Skincare Recommender

**Live app:** LIVE-LINK-HERE

A web app that recommends skincare products for your skin type and skin concerns, explains *why* each product was picked, builds a full morning and night routine, and checks any product from its ingredient list.

## What it does
1. **Your skin:** choose your skin type (dry, oily, combination, normal, sensitive) and any concerns (acne, dark spots, dullness, fine lines, redness, dehydration).
2. **Find products:** the top 5 products of any type (moisturiser, cleanser, serum, mask, eye cream, sunscreen) for your skin, each with a match score and rating.
3. **Why this product?** the ingredients that make it a good match for your skin type, and the ones that help with your concerns.
4. **Build my routine:** the best product for every step of a morning and night routine, with the total cost.
5. **Check any product:** paste the ingredient list of any product, even one that is not in the data, and see which skin types it suits and the most similar products.
6. **Skin insights:** charts showing whether expensive products get better ratings, and which ingredients go with each skin type.

## How it works
- **Data:** 1,472 real Sephora products with their price, rating, full ingredient list and the skin types each one is marked as suitable for. 1,253 have a usable ingredient list.
- **Ingredients to numbers (TF-IDF):** each ingredient list becomes a row of numbers. Common ingredients like water get a small weight; rarer ones get a bigger weight.
- **One model per skin type (logistic regression):** each model learns which ingredients appear more often in products for that skin type. Its output is the **match score**.
- **Explanations:** for any product, the app shows which of its ingredients push the match score up or down (model weight × amount of the ingredient).
- **Concerns:** a list of well-known ingredients for each concern (for example, salicylic acid for acne, vitamin C for dark spots). Products containing them get a small bonus in the ranking.
- **Similar products:** cosine similarity between ingredient lists.

## Results
How often each model is right on products it has **not** seen (5-fold cross-validation, balanced accuracy). Guessing at random would score 50%.

| Skin type | Accuracy |
|---|---|
| Dry | 64% |
| Oily | 64% |
| Combination | 65% |
| Normal | 65% |
| Sensitive | 64% |

**What this means:** the ingredients do carry a real signal about skin type, clearly better than guessing. But the models are far from perfect, mainly because many products are marked as suitable for *every* skin type, which makes the labels fuzzy.

## Limitations
- The skin-type labels come from the retailer, not from dermatologists.
- Ingredient lists show *what* is in a product, not *how much*, so two products with similar lists can still feel different on the skin.
- Prices are in US dollars in the data and are converted at approximate rates.
- This is a guide, not medical advice.

## Run it yourself
```bash
pip3 install -r requirements.txt
python3 model.py        # trains the models and prints the accuracy table
streamlit run app.py    # opens the app
```

## Files
| File | What it does |
|---|---|
| `app.py` | The Streamlit app: all pages, the page-swipe animation and the layout |
| `model.py` | Data cleaning, TF-IDF, the five models, recommendations, routine, explanations |
| `explore.py` | First look at the data |
| `welcome.html` | The animated welcome page (blooming flower, drawn in code) |
| `bubbles.svg` | The serum-bubble background (drawn in code) |
| `data/cosmetics.csv` | The dataset |

## Credits
- Data: [Cosmetics datasets](https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets) on Kaggle (Sephora products).
- Font: Pinyon Script (Google Fonts).
- All graphics were drawn in code for this project.
