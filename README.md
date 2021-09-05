# Webinar: Do you speak NLP? Ein Streifzug durch modernes Natural Language Processing mit Python

Dieses Repo enthält Präsentation, Python Code und Resourcen des Webinars 

_"Do you speak NLP? Ein Streifzug durch modernes Natural Language Processing mit Python"_ 

vom 31. August 2021. Eine Live-Aufzeichnung des Webinars findet sich hier:

https://its-people.de/do-you-speak-nlp

### Präsentation

Die Präsentation basiert auf einem Jupyter Notebook, das mittels des (_genialen_) [RISE Plugins](https://rise.readthedocs.io/en/stable/)
in eine Slideshow (mit interaktiven Elementen) umgewandelt wurde / werden kann.

Das Jupyter Notebook befindet sich im File `presentation.ipynb` und kann auch ohne RISE Plugins über 

```bash
$ jupyter notebook presentation.ipyn
```

verwendet werden. 

Wichtig bei der Verwendung als Slideshow ist, dass sich das CSS-File `presentation.css` im selben Folder wie die 
Präsentation befindet.

### Source Code

Das Folder `src/` enthält den Python-Code für die "Let's Test"-Parts des Webinars. Die Webinar-Parts entsprechen den 
folgenden Python-Scripts:

* "Rule-based NLP" --> `rule_based.py`
* "Statistical NLP" --> `stats_based.py`
* "Neural NLP" --> `nn_based.py`

Jedes dieser Scripts kann als einfaches Python-Script ausgeführt werden, bspw. für Rule-based:

```bash
$ python3 src/rule_based.py
```

Bitte stellen Sie sicher, dass die folgenden Dependencies installiert sind:

| Dependency | Version | Source                           |
| ---        | ---     | ---                              |
| python     | 3.8     | -                                |
| pandas     | 1.3.2   | https://pypi.org/project/pandas/ |
| nltk       | 3.6.2   | https://pypi.org/project/nltk/   |
| gensim     | 4.1.0   | https://pypi.org/project/gensim/ |

(Alle dependencies können im `pyproject.toml`-File gefunden und von dort installiert werden)

### Data 

#### Amazon Reviews Dataset

Die Amazon-Reviews, die für die Tests verwendet wurden, sind **nicht** Teil des Repositorys; das Dataset kann aber hier
heruntergeladen werden:

https://www.kaggle.com/arhamrumi/amazon-product-reviews

Der Download kommt als zip-File "archive.zip". Bitte entpacken Sie das zip-File und kopieren (oder verschieben) Sie
`Reviews.csv` nach `data/`. Bevor das Dataset verwendet werden kann, müssen noch die folgenden Änderungen daran 
vorgenommen werden:

1. Umbenennen: `Reviews.csv` --> `amazon_misc_products_reviews.csv`
2. Die Headline des CSV-Files sollte wie folgt aussehen:

```bash
Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,rating,Time,title,text
```

#### Positive / Negative Words

Unser regelbasiertes NLP-"Modell" verwendet Listen mit positiven und negativen Wörtern, zusammengestellt vom linguist. 
Department der Universität Chicago, zu finden hier:

http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar

#### Most Common Words English

Die Datei `data/en_most_common.txt` enthält eine Liste der 3.000 häufigsten Wörter der Englischen Sprache basierend auf 
verschiedenen, beliebten TV-Shows der letzten Jahre.

Grund für die Wahl dieser eher speziellen "most common words"-Liste ist, dass darin derselbe oder ein zumindest sehr 
ähnlicher, eher informeller Sprachgebrauch abgebildet wird, der auch in dem meisten Amazon-Reviews anzutreffen ist.
