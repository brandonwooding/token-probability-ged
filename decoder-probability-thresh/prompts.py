"""
Prompt templates for the decoder probability-threshold GED experiment.

The method scores the ORIGINAL sentence's tokens (teacher-forced) and reads off
their probabilities. A template only changes the *context* the model sees before
those tokens -- it never changes which tokens get scored.

Contract: a template is a function `(sentence, language) -> prefix`. The caller
builds the full input as `prefix + sentence` and scores only the `sentence`
part. So:

    none       full = sentence
    variant_a  full = <framing> + sentence
    variant_b  full = original:<sentence> ... perfect:<sentence to score>
    fewshot    full = <demos> + original:<sentence> ... perfect:<sentence to score>

For variant_b / fewshot the sentence appears twice: once as context ("original")
and once as the scored tokens ("perfect"). That is intended.

`language` is the MultiGED language folder name ("english", "czech", "german",
"italian", "swedish"), i.e. exactly what paths.language(dataset) returns. Every
piece of prompt wording is written in the target language, so a German sentence
is never framed by English instructions -- the framing has to be in-distribution
for the probabilities it conditions to mean anything.

To tune the experiment you only need to edit the strings in this file: LANGUAGES
holds one block per language, and adding a sixth language means adding one entry.
"""

# ---------------------------------------------------------------------------
# PER-LANGUAGE WORDING -- edit these freely.
#
# Each block supplies:
#   framing          variant_a's single-copy framing ("this text is perfect ...")
#   original_header  the "original sentence:" slot label   (variant_b, fewshot)
#   perfect_header   the "perfect <lang> sentence:" label   (variant_b, fewshot)
#   fewshot          (sentence_with_errors, corrected_sentence) demonstrations,
#                    teaching a BASE model the original->perfect pattern by
#                    example so it shifts toward clean, edited text without
#                    relying on instruction-following. The errors are typical
#                    learner errors for that language (agreement, case, word
#                    order, auxiliary choice, false friends).
# ---------------------------------------------------------------------------
LANGUAGES = {
    "english": {
        "framing": "The following sentence is in perfect English, with no mistakes:\n",
        "original_header": "original sentence:\n",
        "perfect_header": "\n\nperfect english sentence:\n",
        "fewshot": [
            ("I has went to the shop yesterday.", "I went to the shop yesterday."),
            ("She dont like the coffees very much.", "She doesn't like coffee very much."),
            ("We was very exciting about the trip.", "We were very excited about the trip."),
            ("He explain me the problem last week.", "He explained the problem to me last week."),
        ],
    },
    "czech": {
        "framing": "Následující věta je v dokonalé češtině, bez chyb:\n",
        "original_header": "původní věta:\n",
        "perfect_header": "\n\ndokonalá česká věta:\n",
        "fewshot": [
            ("Včera jsem šel do obchod.", "Včera jsem šel do obchodu."),
            ("Ona nemá ráda kávu moc.", "Ona nemá kávu moc ráda."),
            ("Byli jsme velmi vzrušující z výletu.", "Byli jsme z výletu velmi nadšení."),
            ("On mi vysvětlil ten problém minulý týdnu.", "On mi vysvětlil ten problém minulý týden."),
        ],
    },
    "german": {
        "framing": "Der folgende Satz ist in perfektem Deutsch, ohne Fehler:\n",
        "original_header": "ursprünglicher Satz:\n",
        "perfect_header": "\n\nperfekter deutscher Satz:\n",
        "fewshot": [
            ("Ich habe gestern in die Schule gegangen.", "Ich bin gestern in die Schule gegangen."),
            ("Sie mag nicht der Kaffee sehr.", "Sie mag den Kaffee nicht sehr."),
            ("Wir waren sehr aufregend über die Reise.", "Wir waren wegen der Reise sehr aufgeregt."),
            ("Er hat mir das Problem letzten Woche erklärt.", "Er hat mir das Problem letzte Woche erklärt."),
        ],
    },
    "italian": {
        "framing": "La frase seguente è in italiano perfetto, senza errori:\n",
        "original_header": "frase originale:\n",
        "perfect_header": "\n\nfrase italiana perfetta:\n",
        "fewshot": [
            ("Ieri io ho andato al negozio.", "Ieri sono andato al negozio."),
            ("Lei non piace molto il caffè.", "A lei non piace molto il caffè."),
            ("Eravamo molto eccitante per il viaggio.", "Eravamo molto entusiasti del viaggio."),
            ("Lui mi ha spiegato il problema la settimana scorso.", "Lui mi ha spiegato il problema la settimana scorsa."),
        ],
    },
    "swedish": {
        "framing": "Följande mening är på perfekt svenska, utan fel:\n",
        "original_header": "ursprunglig mening:\n",
        "perfect_header": "\n\nperfekt svensk mening:\n",
        "fewshot": [
            ("Jag har gick till affären igår.", "Jag gick till affären igår."),
            ("Hon tycker inte om kaffe mycket.", "Hon tycker inte så mycket om kaffe."),
            ("Vi var mycket spännande inför resan.", "Vi var mycket spända inför resan."),
            ("Han förklarade mig problemet förra veckan.", "Han förklarade problemet för mig förra veckan."),
        ],
    },
    # The three below are C4_200M synthetic-training languages with no MultiGED
    # eval set. They exist so fine-tuning can build variant_b examples for
    # es/ro/ru; the templates themselves are never scored in these languages.
    "spanish": {
        "framing": "La siguiente oración está en español perfecto, sin errores:\n",
        "original_header": "oración original:\n",
        "perfect_header": "\n\noración española perfecta:\n",
        "fewshot": [
            ("Ayer yo he ido a la tienda.", "Ayer fui a la tienda."),
            ("A ella no le gusta mucho el cafés.", "A ella no le gusta mucho el café."),
            ("Estábamos muy emocionante por el viaje.", "Estábamos muy emocionados por el viaje."),
            ("Él me explicó el problema la semana pasado.", "Él me explicó el problema la semana pasada."),
        ],
    },
    "romanian": {
        "framing": "Următoarea propoziție este în română perfectă, fără greșeli:\n",
        "original_header": "propoziție originală:\n",
        "perfect_header": "\n\npropoziție românească perfectă:\n",
        "fewshot": [
            ("Ieri eu am mers la magazin.", "Ieri am mers la magazin."),
            ("Ea nu îi place cafeaua foarte mult.", "Ei nu îi place cafeaua foarte mult."),
            ("Eram foarte entuziasmant pentru călătorie.", "Eram foarte entuziasmați de călătorie."),
            ("El mi-a explicat problema săptămâna trecut.", "El mi-a explicat problema săptămâna trecută."),
        ],
    },
    "russian": {
        "framing": "Следующее предложение написано на идеальном русском языке, без ошибок:\n",
        "original_header": "исходное предложение:\n",
        "perfect_header": "\n\nидеальное русское предложение:\n",
        "fewshot": [
            ("Вчера я пошёл в магазина.", "Вчера я пошёл в магазин."),
            ("Она не любит кофе очень.", "Она не очень любит кофе."),
            ("Мы были очень волнующие из-за поездки.", "Мы были очень взволнованы из-за поездки."),
            ("Он объяснил мне проблему на прошлой неделю.", "Он объяснил мне проблему на прошлой неделе."),
        ],
    },
}


def _wording(language: str) -> dict:
    """Look up one language block, failing loudly on an unsupported language --
    silently falling back to English would quietly invalidate the run."""
    try:
        return LANGUAGES[language]
    except KeyError:
        raise KeyError(
            f"no prompt wording for language {language!r}; "
            f"add a block to LANGUAGES in prompts.py (have: {sorted(LANGUAGES)})"
        ) from None


# ---------------------------------------------------------------------------
# TEMPLATES
# ---------------------------------------------------------------------------
def none(sentence: str, language: str) -> str:
    """Baseline: no prompt, score the raw sentence. Language-independent, but
    takes `language` so every template shares one signature."""
    return ""


def variant_a(sentence: str, language: str) -> str:
    """Single copy of the sentence, framed as perfect text in its language."""
    return _wording(language)["framing"]


def variant_b(sentence: str, language: str) -> str:
    """Two slots: show the sentence as the 'original', then ask for the
    'perfect' version -- the caller teacher-forces the same sentence there."""
    w = _wording(language)
    return f"{w['original_header']}{sentence}{w['perfect_header']}"


def fewshot(sentence: str, language: str) -> str:
    """Few-shot original->corrected demonstrations, then the sentence in the
    'perfect' slot. Conditions a base model toward clean text by example."""
    w = _wording(language)
    demos = "\n\n".join(
        f"{w['original_header']}{original}{w['perfect_header']}{corrected}"
        for original, corrected in w["fewshot"]
    )
    return f"{demos}\n\n{w['original_header']}{sentence}{w['perfect_header']}"


TEMPLATES = {
    "none": none,
    "variant_a": variant_a,
    "variant_b": variant_b,
    "fewshot": fewshot,
}
