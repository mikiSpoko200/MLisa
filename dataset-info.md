

| Entry | value |
|------|------|
| jpg compression ratio | 14.6 |

# Full Dataset (train + test)

Shortest list length in ClassificationTarget.ARTIST: Length = 479
Shortest list length in ClassificationTarget.GENRE: Length = 1902
Shortest list length in ClassificationTarget.STYLE: Length = 98

| target                                                    | class                                           | count |
|-----------------------------------------------------------|-------------------------------------------------|-------|
| style                                                     | Action painting                                 | 98    |
| style                                                     | Analytical Cubism                               | 110   |
| style                                                     | Synthetic Cubism                                | 216   |
| style                                                     | New Realism                                     | 314   |
| style                                                     | Contemporary Realism                            | 481   |
| style                                                     | Pointillism                                     | 513   |
| style                                                     | Fauvism                                         | 934   |
| style                                                     | Ukiyo e                                         | 1167  |
| style                                                     | Mannerism Late Renaissance                      | 1279  |
| style                                                     | Minimalism                                      | 1337  |
| style                                                     | High Renaissance                                | 1343  |
| style                                                     | Early Renaissance                               | 1391  |
| style                                                     | Pop Art                                         | 1483  |
| style                                                     | Color Field Painting                            | 1615  |
| style                                                     | Rococo                                          | 2089  |
| style                                                     | Cubism                                          | 2235  |
| style                                                     | Naive Art Primitivism                           | 2405  |
| style                                                     | Northern Renaissance                            | 2552  |
| style                                                     | Abstract Expressionism                          | 2782  |
| style                                                     | Baroque                                         | 4241  |
| style                                                     | Art Nouveau                                     | 4334  |
| style                                                     | Symbolism                                       | 4528  |
| style                                                     | Post Impressionism                              | 6451  |
| style                                                     | Expressionism                                   | 6736  |
| style                                                     | Romanticism                                     | 7019  |
| style                                                     | Realism                                         | 10733 |
| style                                                     | Impressionism                                   | 13060 |
|   -    |         TOTAL      | 81446 |

| target | class                 | count |
|--------|-----------------------|-------|
| artist | Salvador Dali         | 479   |
| artist | Raphael Kirchner      | 516   |
| artist | Ivan Shishkin         | 520   |
| artist | Ilya Repin            | 539   |
| artist | Childe Hassam         | 550   |
| artist | Eugene Boudin         | 555   |
| artist | Martiros Saryan       | 575   |
| artist | Ivan Aivazovsky       | 577   |
| artist | Paul Cezanne          | 579   |
| artist | Edgar Degas           | 611   |
| artist | Boris Kustodiev       | 633   |
| artist | Gustave Dore          | 753   |
| artist | Pablo Picasso         | 762   |
| artist | Marc Chagall          | 765   |
| artist | Rembrandt             | 777   |
| artist | John Singer Sargent   | 784   |
| artist | Albrecht Durer        | 828   |
| artist | Camille Pissarro      | 887   |
| artist | Pyotr Konchalovsky    | 919   |
| artist | Claude Monet          | 1334  |
| artist | Pierre Auguste Renoir | 1400  |
| artist | Nicholas Roerich      | 1819  |
| artist | Vincent van Gogh      | 1890  |
|   -    |         TOTAL         | 19052 |

| target | class              | count |
|--------|--------------------|-------|
| genre  | illustration       | 1902  |
| genre  | nude painting      | 1923  |
| genre  | still life         | 2788  |
| genre  | sketch and study   | 3943  |
| genre  | cityscape          | 4603  |
| genre  | abstract painting  | 4968  |
| genre  | religious painting | 6538  |
| genre  | genre painting     | 10859 |
| genre  | landscape          | 13358 |
| genre  | portrait           | 14113 |
|   -    |         TOTAL      | 64995 |

```python
import enum


class ClassificationTarget(enum.Enum):
    STYLE = enum.auto()
    ARTIST = enum.auto()
    GENRE = enum.auto()


var = {
    ClassificationTarget.STYLE:
        [('Action painting', 98),
         ('Analytical Cubism', 110),
         ('Synthetic Cubism', 216),
         ('New Realism', 314),
         ('Contemporary Realism', 481),
         ('Pointillism', 513),
         ('Fauvism', 934),
         ('Ukiyo e', 1167),
         ('Mannerism Late Renaissance', 1279),
         ('Minimalism', 1337),
         ('High Renaissance', 1343),
         ('Early Renaissance', 1391),
         ('Pop Art', 1483),
         ('Color Field Painting', 1615),
         ('Rococo', 2089),
         ('Cubism', 2235),
         ('Naive Art Primitivism', 2405),
         ('Northern Renaissance', 2552),
         ('Abstract Expressionism', 2782),
         ('Baroque', 4241),
         ('Art Nouveau', 4334),
         ('Symbolism', 4528),
         ('Post Impressionism', 6451),
         ('Expressionism', 6736),
         ('Romanticism', 7019),
         ('Realism', 10733),
         ('Impressionism', 13060)],
    ClassificationTarget.ARTIST:
        [('Salvador Dali', 479),
         ('Raphael Kirchner', 516),
         ('Ivan Shishkin', 520),
         ('Ilya Repin', 539),
         ('Childe Hassam', 550),
         ('Eugene Boudin', 555),
         ('Martiros Saryan', 575),
         ('Ivan Aivazovsky', 577),
         ('Paul Cezanne', 579),
         ('Edgar Degas', 611),
         ('Boris Kustodiev', 633),
         ('Gustave Dore', 753),
         ('Pablo Picasso', 762),
         ('Marc Chagall', 765),
         ('Rembrandt', 777),
         ('John Singer Sargent', 784),
         ('Albrecht Durer', 828),
         ('Camille Pissarro', 887),
         ('Pyotr Konchalovsky', 919),
         ('Claude Monet', 1334),
         ('Pierre Auguste Renoir', 1400),
         ('Nicholas Roerich', 1819),
         ('Vincent van Gogh', 1890)],
    ClassificationTarget.GENRE:
        [('illustration', 1902),
         ('nude painting', 1923),
         ('still life', 2788),
         ('sketch and study', 3943),
         ('cityscape', 4603),
         ('abstract painting', 4968),
         ('religious painting', 6538),
         ('genre painting', 10859),
         ('landscape', 13358),
         ('portrait', 14113)]
}
```

# Test dateset 

| target | class                 | count |
|--------|-----------------------|-------| 
| artist | Salvador Dali         | 336   |
| artist | Raphael Kirchner      | 362   |
| artist | Ivan Shishkin         | 364   |
| artist | Ilya Repin            | 378   |
| artist | Childe Hassam         | 385   |
| artist | Eugene Boudin         | 389   |
| artist | Martiros Saryan       | 403   |
| artist | Ivan Aivazovsky       | 404   |
| artist | Paul Cezanne          | 406   |
| artist | Edgar Degas           | 428   |
| artist | Boris Kustodiev       | 444   |
| artist | Gustave Dore          | 528   |
| artist | Pablo Picasso         | 534   |
| artist | Marc Chagall          | 536   |
| artist | Rembrandt             | 544   |
| artist | John Singer Sargent   | 549   |
| artist | Albrecht Durer        | 580   |
| artist | Camille Pissarro      | 621   |
| artist | Pyotr Konchalovsky    | 644   |
| artist | Claude Monet          | 934   |
| artist | Pierre Auguste Renoir | 980   |
| artist | Nicholas Roerich      | 1274  |
| artist | Vincent van Gogh      | 1323  |


| target | class                      | count |
|--------|----------------------------|-------| 
| style  | Action painting            | 69    |
| style  | Analytical Cubism          | 77    |
| style  | Synthetic Cubism           | 152   |
| style  | New Realism                | 220   |
| style  | Contemporary Realism       | 337   |
| style  | Pointillism                | 360   |
| style  | Fauvism                    | 654   |
| style  | Ukiyo e                    | 817   |
| style  | Mannerism Late Renaissance | 896   |
| style  | Minimalism                 | 936   |
| style  | High Renaissance           | 941   |
| style  | Early Renaissance          | 974   |
| style  | Pop Art                    | 1039  |
| style  | Color Field Painting       | 1131  |
| style  | Rococo                     | 1463  |
| style  | Cubism                     | 1565  |
| style  | Naive Art Primitivism      | 1684  |
| style  | Northern Renaissance       | 1787  |
| style  | Abstract Expressionism     | 1948  |
| style  | Baroque                    | 2969  |
| style  | Art Nouveau                | 3034  |
| style  | Symbolism                  | 3170  |
| style  | Post Impressionism         | 4516  |
| style  | Expressionism              | 4716  |
| style  | Romanticism                | 4914  |
| style  | Realism                    | 7514  |
| style  | Impressionism              | 9142  |



| target | class              | count |
|--------|--------------------|-------| 
| genre  | illustration       | 1332  |
| genre  | nude painting      | 1347  |
| genre  | still life         | 1952  |
| genre  | sketch and study   | 2761  |
| genre  | cityscape          | 3223  |
| genre  | abstract painting  | 3478  |
| genre  | religious painting | 4577  |
| genre  | genre painting     | 7602  |
| genre  | landscape          | 9351  |
| genre  | portrait           | 9880  |



```python
import enum


class ClassificationTarget(enum.Enum):
    STYLE = enum.auto()
    ARTIST = enum.auto()
    GENRE = enum.auto()


_ = {
    ClassificationTarget.ARTIST: [
        ('Salvador Dali', 336),
        ('Raphael Kirchner', 362),
        ('Ivan Shishkin', 364),
        ('Ilya Repin', 378),
        ('Childe Hassam', 385),
        ('Eugene Boudin', 389),
        ('Martiros Saryan', 403),
        ('Ivan Aivazovsky', 404),
        ('Paul Cezanne', 406),
        ('Edgar Degas', 428),
        ('Boris Kustodiev', 444),
        ('Gustave Dore', 528),
        ('Pablo Picasso', 534),
        ('Marc Chagall', 536),
        ('Rembrandt', 544),
        ('John Singer Sargent', 549),
        ('Albrecht Durer', 580),
        ('Camille Pissarro', 621),
        ('Pyotr Konchalovsky', 644),
        ('Claude Monet', 934),
        ('Pierre Auguste Renoir', 980),
        ('Nicholas Roerich', 1274),
        ('Vincent van Gogh', 1323)
    ],
    ClassificationTarget.STYLE: [
        ('Action painting', 69),
        ('Analytical Cubism', 77),
        ('Synthetic Cubism', 152),
        ('New Realism', 220),
        ('Contemporary Realism', 337),
        ('Pointillism', 360),
        ('Fauvism', 654),
        ('Ukiyo e', 817),
        ('Mannerism Late Renaissance', 896),
        ('Minimalism', 936),
        ('High Renaissance', 941),
        ('Early Renaissance', 974),
        ('Pop Art', 1039),
        ('Color Field Painting', 1131),
        ('Rococo', 1463),
        ('Cubism', 1565),
        ('Naive Art Primitivism', 1684),
        ('Northern Renaissance', 1787),
        ('Abstract Expressionism', 1948),
        ('Baroque', 2969),
        ('Art Nouveau', 3034),
        ('Symbolism', 3170),
        ('Post Impressionism', 4516),
        ('Expressionism', 4716),
        ('Romanticism', 4914),
        ('Realism', 7514),
        ('Impressionism', 9142)
    ],
    ClassificationTarget.GENRE: [
        ('illustration', 1332),
        ('nude painting', 1347),
        ('still life', 1952),
        ('sketch and study', 2761),
        ('cityscape', 3223),
        ('abstract painting', 3478),
        ('religious painting', 4577),
        ('genre painting', 7602),
        ('landscape', 9351),
        ('portrait', 9880)
    ]
}
```

