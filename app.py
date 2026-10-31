"""Skincare Recommender.
Pages: welcome -> skin type -> concerns -> menu -> (find products -> why) / routine / check a product / insights -> end
"""
import base64
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import model as m

st.set_page_config(page_title='Skincare Recommender', page_icon='🌸', layout='centered')

# Script font for page titles, pink tags, rounded buttons everywhere
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pinyon+Script&display=swap');
.script {font-family: 'Pinyon Script', cursive; color: #A8455F; font-size: 2.8rem; line-height: 1.2;}
.chip {display: inline-block; background: #F9CBD2; color: #5A2E38; border-radius: 99px;
       padding: 4px 14px; margin: 3px; font-size: 0.9rem;}
.chip-less {background: #F2F2F2; color: #6B5A5E;}
.chip-help {background: #FCE4E8; border: 1px solid #F1ABBA;}
div[data-testid="stButton"] button {border-radius: 99px;}
</style>
""", unsafe_allow_html=True)

PRODUCT_TYPES = {'Moisturizer': 'Moisturiser', 'Cleanser': 'Cleanser', 'Treatment': 'Treatment / serum',
                 'Face Mask': 'Face mask', 'Eye cream': 'Eye cream', 'Sun protect': 'Sunscreen'}

# Prices in the data are US dollars. Approximate rates from late October 2026.
CURRENCIES = {'Indian rupee (₹)': ('₹', 94.6), 'US dollar ($)': ('$', 1.0),
              'Euro (€)': ('€', 0.89), 'British pound (£)': ('£', 0.75)}

PINK = ['#D88998', '#F1ABBA', '#F2B8C2', '#F9CBD2', '#E5A1AD']

# The order of the pages decides the swipe: a later page comes up from below, an earlier one down from above
PAGES = ['welcome', 'skin', 'concerns', 'menu', 'results', 'why', 'routine', 'check', 'insights', 'end']


@st.cache_resource
def load():
    df = m.load_products()
    vectorizer, X, models = m.build(df)
    return df, vectorizer, X, models


@st.cache_data
def accuracy_table(_X, n_products):
    """Model accuracy on unseen products (takes a few seconds, so it is only worked out once)."""
    return m.evaluate(df, _X)


def go(page, **values):
    """Move to another page, remembering any choices. Also remembers the page we are leaving,
    so it can slide away while the new page slides in."""
    forward = PAGES.index(page) > PAGES.index(st.session_state.page)
    st.session_state.nav = {'from': st.session_state.page, 'forward': forward}
    st.session_state.moves = st.session_state.get('moves', 0) + 1
    st.session_state.page = page
    st.session_state.update(values)


def start_again():
    for key in ['skin', 'concerns', 'label', 'product']:
        st.session_state.pop(key, None)
    go('welcome')


def money(usd):
    symbol, rate = CURRENCIES[st.session_state.get('currency', 'Indian rupee (₹)')]
    return f'{symbol}{usd * rate:,.0f}'


def stars(rating):
    full = round(rating)
    return '★' * full + '☆' * (5 - full) + f'  {rating:.1f}'


def title(text):
    st.markdown(f'<div class="script">{text}</div>', unsafe_allow_html=True)


def chips(names, kind=''):
    st.markdown(''.join(f'<span class="chip {kind}">{name}</span>' for name in names), unsafe_allow_html=True)


def page_style(nav):
    """Floating serum bubbles behind white cards.
    When you move to another page, the old card slides away and the new card slides in,
    and the bubbles move with them, so it feels like the whole page scrolls."""
    svg = (Path(__file__).parent / 'bubbles.svg').read_bytes()
    uri = 'data:image/svg+xml;base64,' + base64.b64encode(svg).decode()
    forward = nav['forward'] if nav else True
    sign = '' if forward else '-'          # forward: new page comes up from below; back: down from above
    moves = st.session_state.get('moves', 0)   # new animation name each move, so the browser replays it
    ease = '1.1s cubic-bezier(.45, .05, .2, 1)'   # slow start, long gentle finish
    move_css = ''
    if nav:
        move_css = f"""
[data-testid="stAppViewContainer"]::before {{animation: bg{moves} {ease};}}
@keyframes bg{moves} {{from {{transform: translateY({sign}100vh);}} to {{transform: none;}}}}
[class*="st-key-page_"] {{animation: arrive {ease};}}
@keyframes arrive {{from {{transform: translateY({sign}100vh) scale(0.97); opacity: 0.4;}}
                    to {{transform: none; opacity: 1;}}}}
[class*="st-key-leaving_"] {{position: absolute; top: 0; left: 0; right: 0; z-index: 1; pointer-events: none;
    animation: leave {ease} forwards;}}
@keyframes leave {{to {{transform: translateY({'-' if forward else ''}100vh) scale(0.97); opacity: 0;
                        visibility: hidden;}}}}
"""
    st.markdown(f"""<style>
[data-testid="stHeader"] {{background: transparent;}}
[data-testid="stAppViewContainer"] {{background: #FCE4E8; overflow-x: hidden;}}
/* the bubbles sit on their own layer, so the graphics card can move them smoothly */
[data-testid="stAppViewContainer"]::before {{content: ""; position: fixed; left: 0; right: 0; top: -100vh;
    height: 300vh; background: url("{uri}"); background-size: 900px auto; pointer-events: none;
    will-change: transform;}}
[data-testid="stMain"], section.main {{position: relative; z-index: 1;}}
[data-testid="stMainBlockContainer"], .block-container {{position: relative;}}
[class*="st-key-page_"], [class*="st-key-leaving_"] {{background: rgba(255, 255, 255, 0.9);
    border-radius: 28px; padding: 2rem 2.5rem 2.5rem; will-change: transform, opacity;}}
{move_css}
</style>""", unsafe_allow_html=True)


# ---------- the pages after the welcome ----------
# ghost=True draws a copy of a page we are leaving (it slides away), with its own widget keys

def skin_page(ghost=False):
    g = 'ghost_' if ghost else ''
    title('Tell us about your skin')
    st.caption('Step 1 of 2')
    skin = st.radio('**My skin type is…**', m.SKIN_TYPES, horizontal=True, key=g + 'skin_choice',
                    index=m.SKIN_TYPES.index(st.session_state.get('skin', 'Dry')))
    with st.expander('Not sure? How to tell your skin type'):
        st.write('Wash your face, wait an hour, then look:\n'
                 '- **Dry:** feels tight, may flake\n'
                 '- **Oily:** shiny all over\n'
                 '- **Combination:** shiny forehead and nose, normal or dry cheeks\n'
                 '- **Normal:** comfortable, not shiny or tight\n'
                 '- **Sensitive:** often stings, itches or turns red with new products')
    currency = st.selectbox('Show prices in', list(CURRENCIES), key=g + 'currency_choice',
                            index=list(CURRENCIES).index(st.session_state.get('currency', 'Indian rupee (₹)')))
    st.write('')
    back, nxt = st.columns(2)
    back.button('←  Back', key=g + 'skin_back', width='stretch', on_click=go, args=('welcome',))
    nxt.button('Next  →', key=g + 'skin_next', type='primary', width='stretch', on_click=go,
               args=('concerns',), kwargs=dict(skin=skin, currency=currency))


def concerns_page(ghost=False):
    g = 'ghost_' if ghost else ''
    title('What would you like to improve?')
    st.caption('Step 2 of 2 · pick as many as you like, or none')
    concerns = st.pills('My skin concerns', list(m.CONCERNS), selection_mode='multi', key=g + 'concern_choice',
                        default=st.session_state.get('concerns', []))
    st.caption('Products that contain ingredients known to help with your concerns move higher in your list.')
    st.write('')
    back, nxt = st.columns(2)
    back.button('←  Back', key=g + 'concerns_back', width='stretch', on_click=go, args=('skin',))
    nxt.button('Done  →', key=g + 'concerns_next', type='primary', width='stretch', on_click=go,
               args=('menu',), kwargs=dict(concerns=list(concerns or [])))


def menu_page(ghost=False):
    g = 'ghost_' if ghost else ''
    skin, concerns = st.session_state.skin, st.session_state.get('concerns', [])
    title('Your skin profile is ready')
    chips([f'{skin} skin'] + concerns)
    st.write('What would you like to do?')
    options = [('results', 'Find products', 'The best products of one type for your skin'),
               ('routine', 'Build my routine', 'A full morning and night routine, with total cost'),
               ('check', 'Check any product', 'Paste an ingredient list and see which skin it suits'),
               ('insights', 'Skin insights', 'What the data says about prices, ratings and ingredients')]
    for row in (options[:2], options[2:]):
        cols = st.columns(2)
        for col, (target, name, text) in zip(cols, row):
            with col.container(border=True):
                st.markdown(f'### {name}')
                st.caption(text)
                st.button('Open  →', key=f'{g}open_{target}', width='stretch', on_click=go, args=(target,))
    st.write('')
    back, end = st.columns(2)
    back.button('←  Change my answers', key=g + 'menu_back', width='stretch', on_click=go, args=('concerns',))
    end.button('Finish  →', key=g + 'menu_end', type='primary', width='stretch', on_click=go, args=('end',))


def results_page(ghost=False):
    g = 'ghost_' if ghost else ''
    skin, concerns = st.session_state.skin, st.session_state.get('concerns', [])
    title(f'Picked for {skin.lower()} skin')
    label = st.radio('I am looking for a…', list(PRODUCT_TYPES), format_func=PRODUCT_TYPES.get, horizontal=True,
                     key=g + 'label_choice',
                     index=list(PRODUCT_TYPES).index(st.session_state.get('label', 'Moisturizer')))
    if not ghost:
        st.session_state.label = label
    top = m.recommend(df, X, models, skin, label, concerns, n=5)
    if top.empty:
        st.info('No products found for this combination. Try another product type.')
    for i, row in top.iterrows():
        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(f"**{row['Name']}**  \n{row['Brand']}")
            left.progress(float(row['match']), text=f"{row['match']:.0%} match for {skin.lower()} skin")
            helped = m.concerns_helped(row['Ingredients'], concerns)
            if helped:
                left.caption('Helps with: ' + ', '.join(helped))
            right.markdown(f"**{money(row['Price'])}**  \n{stars(row['Rank'])}")
            right.button('Why this?', key=f'{g}why{i}', width='stretch', on_click=go, args=('why',),
                         kwargs=dict(product=i, why_from='results'))
    st.button('←  Back to menu', key=g + 'results_back', on_click=go, args=('menu',))


def why_page(ghost=False):
    g = 'ghost_' if ghost else ''
    skin, i = st.session_state.skin, st.session_state.product
    concerns = st.session_state.get('concerns', [])
    row = df.loc[i]
    good, less = m.explain(vectorizer, X, models, skin, i)
    title('Why this product?')
    st.markdown(f"### {row['Name']}\n{row['Brand']} · {money(row['Price'])} · {stars(row['Rank'])}")
    match = models[skin].predict_proba(X[i])[0, 1]
    st.progress(float(match), text=f'{match:.0%} match for {skin.lower()} skin')

    st.markdown(f'**Ingredients that suit {skin.lower()} skin**')
    if len(good):
        chips(good.index)
    else:
        st.write('No single ingredient stands out; the match comes from the overall mix.')
    if len(less):
        st.markdown(f'**Ingredients more common in products not meant for {skin.lower()} skin**')
        chips(less.index, 'chip-less')
    for concern, names in m.concerns_helped(row['Ingredients'], concerns).items():
        st.markdown(f'**Helps with {concern.lower()}**')
        chips(names, 'chip-help')

    with st.expander('How is the match score worked out?'):
        st.write(f'I trained a model on {len(df):,} real products. For each one, the data says which skin '
                 'types it suits. The model learned which ingredients appear more often in products for '
                 f'{skin.lower()} skin. The match score is how strongly this product\'s ingredients point to '
                 f'{skin.lower()} skin. It is a guide based on ingredient patterns, not medical advice: '
                 'patch-test new products and see a dermatologist for skin problems.')
    with st.expander('Full ingredient list'):
        st.write(row['Ingredients'])

    back_to = st.session_state.get('why_from', 'results')
    label = 'Back to my routine' if back_to == 'routine' else 'Back to recommendations'
    st.button(f'←  {label}', key=g + 'why_back', on_click=go, args=(back_to,))


def routine_page(ghost=False):
    g = 'ghost_' if ghost else ''
    skin, concerns = st.session_state.skin, st.session_state.get('concerns', [])
    title('Your routine')
    st.caption(f'The best match for {skin.lower()} skin at every step'
               + (f', favouring {", ".join(c.lower() for c in concerns)}.' if concerns else '.'))
    plan = m.routine(df, X, models, skin, concerns)
    for time, steps in plan.items():
        st.markdown(f"#### {time}")
        for n, (step, i) in enumerate(steps, start=1):
            row = df.loc[i]
            with st.container(border=True):
                left, right = st.columns([3, 1])
                left.markdown(f"**{n}. {step}** · {row['Name']}  \n{row['Brand']} · {stars(row['Rank'])}")
                right.markdown(f"**{money(row['Price'])}**")
                right.button('Why?', key=f'{g}rwhy{time}{i}', width='stretch', on_click=go, args=('why',),
                             kwargs=dict(product=i, why_from='routine'))
    products = {i for steps in plan.values() for _, i in steps}
    st.metric('Total for all products', money(df.loc[list(products), 'Price'].sum()),
              help='Each product counted once, even if you use it morning and night.')
    st.caption('Use sunscreen every morning, and add one new product at a time so you can tell what suits you.')
    st.button('←  Back to menu', key=g + 'routine_back', on_click=go, args=('menu',))


EXAMPLE = ('Water, Niacinamide, Pentylene Glycol, Zinc PCA, Dimethyl Isosorbide, Tamarindus Indica Seed Gum, '
           'Xanthan Gum, Isoceteth-20, Ethoxydiglycol, Phenoxyethanol, Chlorphenesin')


def check_page(ghost=False):
    g = 'ghost_' if ghost else ''
    title('Check any product')
    st.caption('Copy the ingredient list from the box or the brand\'s website and paste it here. '
               'Works for brands that are not in my data too.')
    if not ghost and st.button('Try an example serum', key='example'):
        st.session_state.pasted = EXAMPLE
    text = st.text_area('Ingredients (separated by commas)', key=g + 'pasted', height=120)
    if text.strip() and not ghost:
        fit, similar, known = m.check_product(text, vectorizer, X, models, df)
        if known == 0:
            st.warning('I did not recognise any of these ingredients. Check that they are separated by commas.')
        else:
            st.markdown('**How well it suits each skin type**')
            for skin, p in sorted(fit.items(), key=lambda kv: -kv[1]):
                st.progress(float(p), text=f'{skin}: {p:.0%}')
            helped = m.concerns_helped(text, list(m.CONCERNS))
            if helped:
                st.markdown('**Contains ingredients known to help with**')
                chips([f'{c}: {", ".join(names)}' for c, names in helped.items()], 'chip-help')
            st.markdown('**Most similar products in my data**')
            for _, row in similar.iterrows():
                st.write(f"• {row['Brand']} – {row['Name']} ({row['similarity']:.0%} similar, {money(row['Price'])})")
    st.button('←  Back to menu', key=g + 'check_back', on_click=go, args=('menu',))


def insights_page(ghost=False):
    g = 'ghost_' if ghost else ''
    title('Skin insights')
    st.caption(f'What {len(df):,} real skincare products tell us')

    st.markdown('#### Do expensive products get better ratings?')
    prices = m.price_vs_rating(df).reset_index()
    prices.columns = ['Price group', 'Average rating']
    chart = (alt.Chart(prices).mark_bar(color=PINK[0], cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
             .encode(x=alt.X('Price group:N', sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                     y=alt.Y('Average rating:Q', scale=alt.Scale(domain=[0, 5]), title='Average rating (out of 5)'),
                     tooltip=['Price group', 'Average rating']).properties(height=260))
    st.altair_chart(chart, width='stretch')
    cheap, dear = prices['Average rating'].iloc[0], prices['Average rating'].iloc[-1]
    if abs(dear - cheap) < 0.2:
        st.write(f'**Not really.** The cheapest products average {cheap:.2f} stars and the most expensive '
                 f'{dear:.2f}. Paying more does not buy a much better-rated product.')
    elif dear > cheap:
        st.write(f'**A little.** The most expensive products average {dear:.2f} stars against {cheap:.2f} '
                 'for the cheapest.')
    else:
        st.write(f'**No, the opposite.** The cheapest products average {cheap:.2f} stars and the most '
                 f'expensive only {dear:.2f}.')

    st.markdown('#### Which ingredients go with each skin type?')
    skin = st.radio('Skin type', m.SKIN_TYPES, horizontal=True, key=g + 'insight_skin', label_visibility='collapsed')
    top = m.top_ingredients(vectorizer, models, skin).reset_index()
    top.columns = ['Ingredient', 'Weight']
    chart = (alt.Chart(top).mark_bar(color=PINK[1], cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
             .encode(y=alt.Y('Ingredient:N', sort='-x', title=None, axis=alt.Axis(labelLimit=260)),
                     x=alt.X('Weight:Q', title=f'How strongly the model links it to {skin.lower()} skin'),
                     tooltip=['Ingredient', alt.Tooltip('Weight:Q', format='.2f')]).properties(height=280))
    st.altair_chart(chart, width='stretch')

    st.markdown('#### How accurate is the model?')
    if not ghost:
        st.dataframe(accuracy_table(X, len(df)), hide_index=True, width='stretch')
    st.caption('Tested on products the model had not seen. Guessing at random would score 50%. '
               'Many products are marked as suitable for every skin type, which limits how accurate '
               'any model can be on this data.')
    st.button('←  Back to menu', key=g + 'insights_back', on_click=go, args=('menu',))


def end_page(ghost=False):
    g = 'ghost_' if ghost else ''
    skin, concerns = st.session_state.get('skin', ''), st.session_state.get('concerns', [])
    st.markdown('<div style="text-align:center;font-size:4rem">🌸</div>', unsafe_allow_html=True)
    st.markdown('<div class="script" style="text-align:center">Thank you for visiting</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center">Your profile: <b>{skin.lower()} skin</b>'
                + (f' · {", ".join(c.lower() for c in concerns)}' if concerns else '') + '</p>',
                unsafe_allow_html=True)
    st.markdown('<p style="text-align:center">Be gentle with your skin, wear sunscreen every day, '
                'and give new products a few weeks to work.</p>', unsafe_allow_html=True)
    st.caption('This app is a guide based on ingredient patterns in real products, not medical advice. '
               'Data: Sephora products dataset (Kaggle). Built with Python, scikit-learn and Streamlit.')
    st.write('')
    back, again = st.columns(2)
    back.button('←  Back to menu', key=g + 'end_back', width='stretch', on_click=go, args=('menu',))
    again.button('Start again  ↺', key=g + 'end_again', type='primary', width='stretch', on_click=start_again)


PAGE_FUNCTIONS = {'skin': skin_page, 'concerns': concerns_page, 'menu': menu_page, 'results': results_page,
                  'why': why_page, 'routine': routine_page, 'check': check_page, 'insights': insights_page,
                  'end': end_page}

# ---------- page 1: welcome (the blooming flower) ----------
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
page = st.session_state.page

if page == 'welcome':
    st.session_state.pop('nav', None)
    st.markdown((Path(__file__).parent / 'welcome.html').read_text(encoding='utf-8'), unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 2, 1])
    middle.button('Continue  →', type='primary', width='stretch', on_click=go, args=('skin',))
    st.stop()

df, vectorizer, X, models = load()

nav = st.session_state.pop('nav', None)     # set only right after a button click
page_style(nav)
if nav and nav['from'] in PAGE_FUNCTIONS:
    with st.container(key=f"leaving_{nav['from']}"):     # the old page, sliding away
        PAGE_FUNCTIONS[nav['from']](ghost=True)
with st.container(key=f'page_{page}'):                   # the new page, sliding in
    PAGE_FUNCTIONS[page]()
