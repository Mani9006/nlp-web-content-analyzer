---
title: "Lexical and Affective Profiling of Web Content"
subtitle: "An NLP pipeline combining VADER sentiment, RAKE keywords, and Flesch readability over the Common Crawl News subset"
shorttitle: "Lexical and Affective Profiling of Web Content"
year: "2026"
---


# Abstract

Editorial teams, brand managers, and SEO practitioners need fast, explainable summaries of how web pages read and how they feel. Black-box transformer pipelines deliver state-of-the-art accuracy but are expensive to operate and difficult to audit. This study evaluates whether a lightweight, lexicon-based pipeline can replicate transformer-tier decisions on three orthogonal axes: sentiment polarity, keyphrase salience, and reading difficulty. We assemble a 12,000-document evaluation corpus drawn from Common Crawl News (a 50-language English subset) and pair each document with weak labels harvested from publisher-provided structured data. The resulting pipeline reaches a Pearson correlation of 0.81 with a fine-tuned RoBERTa sentiment model on an out-of-distribution news subset while running 47x faster on CPU. RAKE keyphrase extraction recovers 73% of editor-supplied tags within the top-10 list. Flesch-Kincaid grade level on the same corpus correlates 0.94 with the FKGL reference implementation. We argue this combination is the appropriate default for production ingestion pipelines that need explainable scores, deterministic outputs, and predictable cost.

**Keywords:** natural language processing, sentiment analysis, keyphrase extraction, readability, Common Crawl

# Introduction

Web content moderation, brand monitoring, and editorial quality control all require lightweight, explainable signals over arbitrary HTML inputs. Modern transformer models can deliver high accuracy on each task in isolation, but production teams routinely report two practical pain points: per-document inference cost (USD 0.0002-0.002 per call on hosted APIs) and the inability of stakeholders to inspect why a particular document was scored a particular way. Lexicon and rule-based methods remain attractive because they are fast, deterministic, and fully explainable, but they have a reputation for being inaccurate.

## Research Problem

The research problem is whether a carefully assembled lexicon-based pipeline (VADER for sentiment, RAKE for keyphrases, Flesch-Kincaid for readability) can match the practical decision quality of a transformer stack on real-world web content while remaining auditable and cheap to operate. We focus on real-world web content (not curated review datasets) because that is the deployment surface that matters in practice.

## Research Questions and Hypotheses

**Research question:** How does VADER sentiment correlate with a fine-tuned transformer on out-of-distribution news content?

*Hypothesis:* We expect a Pearson correlation above 0.75, sufficient for editorial triage decisions, given VADER's lexicon coverage of news-style language.

**Research question:** What fraction of editor-supplied tags does RAKE recover in its top-10 keyphrase list?

*Hypothesis:* We hypothesize at least 60% recall at top-10, with the remainder being entity-level tags better served by NER post-processing.

**Research question:** Does Flesch-Kincaid grade level computed on extracted body text replicate the reference implementation within 0.5 grade points on average?

*Hypothesis:* We expect mean absolute deviation under 0.5 grade levels and Pearson correlation above 0.9, modulo divergence on very short pages.

**Research question:** How do operational costs (latency, cost-per-1k-pages) compare between the lexicon stack and a hosted transformer API?

*Hypothesis:* We expect at least one order of magnitude reduction in both latency and cost without a corresponding drop in editorial-decision quality.


# Literature Review

## Theories Grounding the Problem

1. **Compositional Lexicon Theory (Hutto & Gilbert, 2014)** — Sentiment intensity can be reconstructed by combining word-level scores with hand-curated rules for negation, intensification, and contrastive conjunctions; this approach is deterministic and inspectable. (Hutto & Gilbert (2014))

2. **RAKE Co-occurrence Theory (Rose et al., 2010)** — Salient keyphrases tend to be sequences of content words that co-occur frequently relative to their individual frequencies; the algorithm operates entirely on within-document statistics, requiring no training corpus. (Rose, Engel, Cramer, & Cowley (2010))

3. **Readability as a Function of Surface Features (Flesch, 1948)** — Reading difficulty correlates with mean sentence length and mean syllables per word; the formula is empirically calibrated against grade-level reading tests and remains the de facto standard in public-sector communication guidelines. (Flesch (1948))

4. **Distributional Hypothesis (Harris, 1954)** — Words that occur in similar contexts tend to have similar meanings; this underpins both lexicon construction and modern embeddings, and motivates the choice of bag-of-words features for tasks where syntactic order matters less than topical content. (Harris (1954))

5. **Editorial Triage Cost Model** — In high-volume content operations, the cost of scoring all incoming documents dominates the cost of scoring them precisely; a screening tier with 10x lower cost and modestly lower accuracy can outperform a single high-accuracy stage on total decision quality. (internal industrial framing)


## Supporting Examples

- BBC and Guardian both publish editor-supplied tags as JSON-LD; this provides ground-truth supervision for keyphrase extraction without manual annotation.
- The U.S. Plain Writing Act of 2010 mandates Flesch-Kincaid-style readability checks on federal communications, demonstrating institutional reliance on lexicon-based readability scores.
- Brandwatch and Talkwalker, both commercial monitoring platforms, deploy hybrid stacks: a lexicon screening tier feeds a transformer escalation tier only on flagged content.

# Research Method

Documents are ingested via a BeautifulSoup parser with content extraction using readability-lxml, producing a clean body text per page. The text is passed through a NLTK tokenizer and POS tagger. Sentiment is computed with VADER over both whole-document and per-sentence views; the per-sentence view yields a polarity histogram that surfaces editorial conflict. RAKE extracts candidate keyphrases scored by degree-over-frequency; the top-10 are returned. Flesch-Kincaid grade level is computed with the standard syllable-counting algorithm; we report grade level, reading ease, and the Gunning Fog index for triangulation. The pipeline's outputs are compared against transformer reference labels (RoBERTa-large fine-tuned on tweet_eval) and editor-supplied tags using Pearson correlation, top-k recall, and absolute deviation.

# Data Description

**Source:** Common Crawl News English subset (12,047 articles) — https://commoncrawl.org/2016/10/news-dataset-available/

**Coverage:** 12,047 English-language news articles, balanced across 38 publishers, sampled from CC-NEWS 2024 dumps

**Schema (selected fields):**

  - url, publisher, publish_date
  - title, body_text (extracted via readability-lxml)
  - editor_tags (when published as JSON-LD)
  - char_count, sentence_count, word_count

**Preprocessing:** We removed boilerplate (navigation, footer, share buttons) using readability-lxml. Articles under 200 words or above 8,000 words were excluded to keep the analysis focused on standard editorial pieces. Where editor_tags were present we normalized them via lowercase and stop-word stripping for fair comparison against extracted phrases.

**License / availability:** Common Crawl content is free for research use under fair-use; downstream redistribution restricted to publisher terms.

# Analysis

## Sentiment correlation against transformer reference

We computed VADER compound score per document and compared against the RoBERTa fine-tuned model. Correlation is reported overall and stratified by article length tertile.

| Subset | n | Pearson r | Spearman rho | Mean abs deviation |
| --- | --- | --- | --- | --- |
| All | 12,047 | 0.812 | 0.794 | 0.097 |
| Short (<400w) | 3,512 | 0.785 | 0.768 | 0.119 |
| Medium (400-1200w) | 5,891 | 0.829 | 0.811 | 0.084 |
| Long (>1200w) | 2,644 | 0.804 | 0.787 | 0.103 |


## Keyphrase recovery against editor tags

Top-10 RAKE keyphrases were compared against editor-supplied JSON-LD tags. We report micro-averaged recall at varying k.

| k | Articles with tags | Recall@k | Precision@k |
| --- | --- | --- | --- |
| 1 | 4,103 | 0.32 | 0.61 |
| 3 | 4,103 | 0.54 | 0.45 |
| 5 | 4,103 | 0.66 | 0.36 |
| 10 | 4,103 | 0.73 | 0.24 |


## Readability calibration

Flesch-Kincaid grade level was computed and compared against textstat reference. Pearson correlation is 0.94 (n=12,047). Mean absolute deviation 0.31 grade points; the largest deviations occur on articles with extensive direct-quote passages.

## Operational cost

Throughput was measured on a single 8-core CPU. The lexicon pipeline averaged 1,820 articles/second; the transformer reference averaged 39 articles/second on the same hardware. End-to-end cost per million articles dropped from USD 217 (transformer) to USD 4.60 (lexicon).


# Discussion

The lexicon pipeline reaches editorial decision quality on sentiment (r=0.81) and grade level (r=0.94) at a small fraction of the operational cost of a transformer stack. Top-10 keyphrase recall of 0.73 is actionable for SEO and tagging workflows but leaves a 27% gap that is concentrated on entity-level phrases. We recommend deploying a named-entity recognizer alongside RAKE rather than replacing RAKE wholesale. The most consequential weakness of the lexicon stack appears on short articles (<400 words), where surface-feature noise dominates and sentiment correlation drops below 0.79.

# Conclusion

A lexicon and rule-based content analysis pipeline is competitive with transformer baselines on three editorial axes — sentiment, keyphrase, readability — at 47x lower latency and an order-of-magnitude lower cost. The deterministic, explainable outputs are an additional asset for stakeholders who need to audit individual decisions. We recommend this stack as the default screening tier in production content ingestion pipelines, with a transformer escalation tier reserved for documents flagged on confidence boundaries.

# Future Work

- Replace RAKE with a hybrid RAKE-plus-NER stack to close the keyphrase recall gap on entity-level tags.
- Investigate domain-adapted lexicons for financial and medical content where general-purpose lexicons under-perform.
- Add a neural reranker as a thin escalation tier on flagged documents while preserving the deterministic-screening contract.
- Publish the evaluation corpus and weak-label harvest as a reproducible benchmark.

# References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly. https://www.nltk.org/book/

2. Hutto, C. J. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* ICWSM-14. https://ojs.aaai.org/index.php/ICWSM/article/view/14550

3. Flesch, R. (1948). *A new readability yardstick.* Journal of Applied Psychology, 32(3), 221-233.

4. Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). *Automatic Keyword Extraction from Individual Documents.* Text Mining: Applications and Theory, Wiley. https://onlinelibrary.wiley.com/doi/10.1002/9780470689646.ch1

5. Harris, Z. S. (1954). *Distributional Structure.* Word 10(2-3), 146-162.

6. Loper, E. & Bird, S. (2002). *NLTK: The Natural Language Toolkit.* Proceedings of the ACL Workshop on Effective Tools and Methodologies for Teaching NLP. https://aclanthology.org/W02-0109/

7. Common Crawl Foundation. *CC-News dataset.* https://commoncrawl.org/2016/10/news-dataset-available/
