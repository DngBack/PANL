"""Tier-1 controlled facts: capitals, currencies, authors, inventors.

Every fact must be *functional* within its family: the subject has exactly one correct
answer entity. That property is what makes a crossed cell genuinely incorrect, so it is
the load-bearing curation rule here, not a stylistic one. Subjects with multiple defensible
answers are deliberately absent (the Netherlands and Bolivia have split capitals, Bulgaria
and Croatia moved to the euro, the radio and the transistor have contested inventors).

Where a subject legitimately admits a second correct answer, list it in `also_correct` and
the block builder will refuse to pair it against the fact that owns that answer.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Final

RELATION_FAMILIES: Final[tuple[str, ...]] = (
    "capital_of",
    "currency_of",
    "author_of",
    "inventor_of",
)


#: Symbols that carry identity. Dropping them would fold "C++" onto "C" and hand two
#: distinct facts the same question_id.
_SYMBOL_WORDS: Final[dict[str, str]] = {"+": " plus ", "#": " sharp ", "&": " and "}


def slugify(text: str) -> str:
    """ASCII, lowercase, hyphenated identifier. Stable across runs; used to build ids."""
    for symbol, word in _SYMBOL_WORDS.items():
        text = text.replace(symbol, word)
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    out = [ch.lower() if ch.isalnum() else "-" for ch in ascii_text]
    return "-".join(filter(None, "".join(out).split("-")))


@dataclass(frozen=True, slots=True)
class Fact:
    """A single (subject, answer) fact inside one relation family."""

    relation_family: str
    subject: str
    #: Surface form used inside question templates, e.g. "the telephone" for "telephone".
    subject_phrase: str
    answer: str
    #: Answer slugs that are *also* correct for this subject. Normally empty.
    also_correct: tuple[str, ...] = field(default=())

    @property
    def subject_slug(self) -> str:
        return slugify(self.subject)

    @property
    def answer_slug(self) -> str:
        return slugify(self.answer)

    @property
    def fact_id(self) -> str:
        return f"{self.relation_family}:{self.subject_slug}"

    @property
    def question_id(self) -> str:
        """Identity of the question. Family-scoped: `capital_of:france` is not the same
        question identity as `currency_of:france`, and they may fall in different splits."""
        return f"q:{self.relation_family}:{self.subject_slug}"

    @property
    def answer_id(self) -> str:
        return f"a:{self.relation_family}:{self.answer_slug}"

    def correct_answer_ids(self) -> frozenset[str]:
        prefix = f"a:{self.relation_family}:"
        return frozenset({self.answer_id, *(prefix + slug for slug in self.also_correct)})


def _facts(family: str, pairs: tuple[tuple[str, str], ...]) -> tuple[Fact, ...]:
    """Build facts whose template phrase is the subject itself (capitals, currencies)."""
    return tuple(
        Fact(relation_family=family, subject=s, subject_phrase=s, answer=a) for s, a in pairs
    )


def _phrased_facts(family: str, triples: tuple[tuple[str, str, str], ...]) -> tuple[Fact, ...]:
    """Build facts that need a distinct template phrase, e.g. "the telephone"."""
    return tuple(
        Fact(relation_family=family, subject=s, subject_phrase=p, answer=a) for s, p, a in triples
    )


CAPITALS: Final[tuple[Fact, ...]] = _facts(
    "capital_of",
    (
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Brazil", "Brasília"),
        ("Egypt", "Cairo"),
        ("Kenya", "Nairobi"),
        ("Norway", "Oslo"),
        ("Portugal", "Lisbon"),
        ("Greece", "Athens"),
        ("Turkey", "Ankara"),
        ("Vietnam", "Hanoi"),
        ("Thailand", "Bangkok"),
        ("Peru", "Lima"),
        ("Chile", "Santiago"),
        ("Argentina", "Buenos Aires"),
        ("Mexico", "Mexico City"),
        ("Poland", "Warsaw"),
        ("Hungary", "Budapest"),
        ("Austria", "Vienna"),
        ("Sweden", "Stockholm"),
        ("Finland", "Helsinki"),
        ("Denmark", "Copenhagen"),
        ("Ireland", "Dublin"),
        ("Morocco", "Rabat"),
        ("Nigeria", "Abuja"),
        ("Ethiopia", "Addis Ababa"),
        ("Ghana", "Accra"),
        ("Cuba", "Havana"),
        ("Iceland", "Reykjavík"),
        ("Nepal", "Kathmandu"),
        ("Pakistan", "Islamabad"),
        ("Bangladesh", "Dhaka"),
        ("the Philippines", "Manila"),
        ("South Korea", "Seoul"),
        ("Ukraine", "Kyiv"),
        ("Romania", "Bucharest"),
        ("Croatia", "Zagreb"),
        ("Serbia", "Belgrade"),
        ("Switzerland", "Bern"),
        ("Belgium", "Brussels"),
        ("Spain", "Madrid"),
        ("Italy", "Rome"),
        ("Germany", "Berlin"),
        ("Russia", "Moscow"),
        ("India", "New Delhi"),
        ("China", "Beijing"),
        ("Iran", "Tehran"),
        ("Iraq", "Baghdad"),
        ("Saudi Arabia", "Riyadh"),
        ("Colombia", "Bogotá"),
        ("Venezuela", "Caracas"),
        ("Uruguay", "Montevideo"),
        ("Paraguay", "Asunción"),
        ("Ecuador", "Quito"),
        ("Cambodia", "Phnom Penh"),
        ("Laos", "Vientiane"),
        ("Mongolia", "Ulaanbaatar"),
        ("Uzbekistan", "Tashkent"),
        ("Afghanistan", "Kabul"),
        ("New Zealand", "Wellington"),
        ("Tanzania", "Dodoma"),
        ("Senegal", "Dakar"),
        ("Tunisia", "Tunis"),
        ("Algeria", "Algiers"),
        ("Jordan", "Amman"),
        ("Lebanon", "Beirut"),
        ("Qatar", "Doha"),
        ("Oman", "Muscat"),
        ("Czechia", "Prague"),
        ("Slovakia", "Bratislava"),
        ("Slovenia", "Ljubljana"),
        ("Lithuania", "Vilnius"),
        ("Latvia", "Riga"),
        ("Estonia", "Tallinn"),
        ("Belarus", "Minsk"),
        ("Armenia", "Yerevan"),
        ("Azerbaijan", "Baku"),
    ),
)

CURRENCIES: Final[tuple[Fact, ...]] = _facts(
    "currency_of",
    (
        ("Japan", "Japanese yen"),
        ("the United Kingdom", "Pound sterling"),
        ("India", "Indian rupee"),
        ("China", "Chinese yuan"),
        ("Russia", "Russian ruble"),
        ("Poland", "Polish zloty"),
        ("Sweden", "Swedish krona"),
        ("Norway", "Norwegian krone"),
        ("Denmark", "Danish krone"),
        ("Switzerland", "Swiss franc"),
        ("Turkey", "Turkish lira"),
        ("Israel", "Israeli shekel"),
        ("Thailand", "Thai baht"),
        ("Vietnam", "Vietnamese dong"),
        ("South Korea", "South Korean won"),
        ("Brazil", "Brazilian real"),
        ("Mexico", "Mexican peso"),
        ("Argentina", "Argentine peso"),
        ("Chile", "Chilean peso"),
        ("Peru", "Peruvian sol"),
        ("Colombia", "Colombian peso"),
        ("Canada", "Canadian dollar"),
        ("Australia", "Australian dollar"),
        ("New Zealand", "New Zealand dollar"),
        ("South Africa", "South African rand"),
        ("Nigeria", "Nigerian naira"),
        ("Kenya", "Kenyan shilling"),
        ("Egypt", "Egyptian pound"),
        ("Morocco", "Moroccan dirham"),
        ("Ghana", "Ghanaian cedi"),
        ("Ethiopia", "Ethiopian birr"),
        ("Indonesia", "Indonesian rupiah"),
        ("Malaysia", "Malaysian ringgit"),
        ("the Philippines", "Philippine peso"),
        ("Pakistan", "Pakistani rupee"),
        ("Bangladesh", "Bangladeshi taka"),
        ("Sri Lanka", "Sri Lankan rupee"),
        ("Iran", "Iranian rial"),
        ("Saudi Arabia", "Saudi riyal"),
        ("Czechia", "Czech koruna"),
        ("Hungary", "Hungarian forint"),
        ("Romania", "Romanian leu"),
        ("Ukraine", "Ukrainian hryvnia"),
        ("Iceland", "Icelandic króna"),
        ("Singapore", "Singapore dollar"),
        ("Kazakhstan", "Kazakhstani tenge"),
        ("Nepal", "Nepalese rupee"),
        ("Cambodia", "Cambodian riel"),
        ("Mongolia", "Mongolian tugrik"),
        ("Costa Rica", "Costa Rican colón"),
        ("Guatemala", "Guatemalan quetzal"),
        ("Honduras", "Honduran lempira"),
        ("Nicaragua", "Nicaraguan córdoba"),
        ("Paraguay", "Paraguayan guaraní"),
        ("Uruguay", "Uruguayan peso"),
        ("Jamaica", "Jamaican dollar"),
        ("Albania", "Albanian lek"),
        ("Serbia", "Serbian dinar"),
        ("Georgia", "Georgian lari"),
        ("Armenia", "Armenian dram"),
        ("Azerbaijan", "Azerbaijani manat"),
        ("Uzbekistan", "Uzbekistani som"),
        ("Iraq", "Iraqi dinar"),
        ("Jordan", "Jordanian dinar"),
        ("Kuwait", "Kuwaiti dinar"),
        ("Qatar", "Qatari riyal"),
        ("Oman", "Omani rial"),
        ("Tunisia", "Tunisian dinar"),
        ("Algeria", "Algerian dinar"),
        ("Tanzania", "Tanzanian shilling"),
        ("Uganda", "Ugandan shilling"),
        ("Zambia", "Zambian kwacha"),
        ("Botswana", "Botswana pula"),
        ("Angola", "Angolan kwanza"),
        ("Mozambique", "Mozambican metical"),
        ("Laos", "Lao kip"),
    ),
)

AUTHORS: Final[tuple[Fact, ...]] = _facts(
    "author_of",
    (
        ("Pride and Prejudice", "Jane Austen"),
        ("Nineteen Eighty-Four", "George Orwell"),
        ("Moby-Dick", "Herman Melville"),
        ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("To Kill a Mockingbird", "Harper Lee"),
        ("Wuthering Heights", "Emily Brontë"),
        ("Jane Eyre", "Charlotte Brontë"),
        ("Crime and Punishment", "Fyodor Dostoevsky"),
        ("War and Peace", "Leo Tolstoy"),
        ("Don Quixote", "Miguel de Cervantes"),
        ("Ulysses", "James Joyce"),
        ("Mrs Dalloway", "Virginia Woolf"),
        ("The Trial", "Franz Kafka"),
        ("One Hundred Years of Solitude", "Gabriel García Márquez"),
        ("The Old Man and the Sea", "Ernest Hemingway"),
        ("Brave New World", "Aldous Huxley"),
        ("Fahrenheit 451", "Ray Bradbury"),
        ("Dracula", "Bram Stoker"),
        ("Frankenstein", "Mary Shelley"),
        ("The Hobbit", "J. R. R. Tolkien"),
        ("Great Expectations", "Charles Dickens"),
        ("Les Misérables", "Victor Hugo"),
        ("Madame Bovary", "Gustave Flaubert"),
        ("The Stranger", "Albert Camus"),
        ("Lolita", "Vladimir Nabokov"),
        ("Catch-22", "Joseph Heller"),
        ("The Catcher in the Rye", "J. D. Salinger"),
        ("Beloved", "Toni Morrison"),
        ("Things Fall Apart", "Chinua Achebe"),
        ("Middlemarch", "George Eliot"),
        ("The Divine Comedy", "Dante Alighieri"),
        ("Faust", "Johann Wolfgang von Goethe"),
        ("The Picture of Dorian Gray", "Oscar Wilde"),
        ("Heart of Darkness", "Joseph Conrad"),
        ("The Sound and the Fury", "William Faulkner"),
        ("Slaughterhouse-Five", "Kurt Vonnegut"),
        ("On the Road", "Jack Kerouac"),
        ("Invisible Man", "Ralph Ellison"),
        ("The Handmaid's Tale", "Margaret Atwood"),
        ("Norwegian Wood", "Haruki Murakami"),
        ("The Name of the Rose", "Umberto Eco"),
        ("Blindness", "José Saramago"),
        ("Doctor Zhivago", "Boris Pasternak"),
        ("Lord of the Flies", "William Golding"),
        ("The Grapes of Wrath", "John Steinbeck"),
        ("The Master and Margarita", "Mikhail Bulgakov"),
        ("Waiting for the Barbarians", "J. M. Coetzee"),
        ("The Remains of the Day", "Kazuo Ishiguro"),
        ("Midnight's Children", "Salman Rushdie"),
        ("The Bell Jar", "Sylvia Plath"),
    ),
)

INVENTORS: Final[tuple[Fact, ...]] = _phrased_facts(
    "inventor_of",
    (
        ("telephone", "the telephone", "Alexander Graham Bell"),
        ("phonograph", "the phonograph", "Thomas Edison"),
        ("dynamite", "dynamite", "Alfred Nobel"),
        ("World Wide Web", "the World Wide Web", "Tim Berners-Lee"),
        ("Linux kernel", "the Linux kernel", "Linus Torvalds"),
        ("Python programming language", "the Python programming language", "Guido van Rossum"),
        ("C programming language", "the C programming language", "Dennis Ritchie"),
        ("C++ programming language", "the C++ programming language", "Bjarne Stroustrup"),
        ("Ruby programming language", "the Ruby programming language", "Yukihiro Matsumoto"),
        ("Perl programming language", "the Perl programming language", "Larry Wall"),
        ("PHP programming language", "the PHP programming language", "Rasmus Lerdorf"),
        ("JavaScript programming language", "the JavaScript programming language", "Brendan Eich"),
        ("Java programming language", "the Java programming language", "James Gosling"),
        ("Lisp programming language", "the Lisp programming language", "John McCarthy"),
        ("Fortran programming language", "the Fortran programming language", "John Backus"),
        ("Pascal programming language", "the Pascal programming language", "Niklaus Wirth"),
        ("Erlang programming language", "the Erlang programming language", "Joe Armstrong"),
        ("Ethernet", "Ethernet", "Robert Metcalfe"),
        ("Rubik's Cube", "the Rubik's Cube", "Ernő Rubik"),
        ("braille writing system", "the braille writing system", "Louis Braille"),
        ("Morse code", "Morse code", "Samuel Morse"),
        ("cotton gin", "the cotton gin", "Eli Whitney"),
        ("diesel engine", "the diesel engine", "Rudolf Diesel"),
        ("revolver", "the revolver", "Samuel Colt"),
        ("microwave oven", "the microwave oven", "Percy Spencer"),
        ("zipper", "the zipper", "Gideon Sundback"),
        ("ballpoint pen", "the ballpoint pen", "László Bíró"),
        ("Kevlar", "Kevlar", "Stephanie Kwolek"),
        ("nylon", "nylon", "Wallace Carothers"),
        ("Teflon", "Teflon", "Roy Plunkett"),
        ("air conditioning", "air conditioning", "Willis Carrier"),
        ("vulcanized rubber", "vulcanized rubber", "Charles Goodyear"),
        ("stethoscope", "the stethoscope", "René Laennec"),
        ("hovercraft", "the hovercraft", "Christopher Cockerell"),
        ("safety elevator", "the safety elevator", "Elisha Otis"),
        ("Gatling gun", "the Gatling gun", "Richard Gatling"),
    ),
)

TIER1_FACTS: Final[tuple[Fact, ...]] = CAPITALS + CURRENCIES + AUTHORS + INVENTORS


def facts_by_family(families: tuple[str, ...] = RELATION_FAMILIES) -> dict[str, list[Fact]]:
    """Group the Tier-1 facts by family, in declaration order, for the requested families."""
    unknown = set(families) - set(RELATION_FAMILIES)
    if unknown:
        msg = f"unknown relation families: {sorted(unknown)}"
        raise ValueError(msg)
    grouped: dict[str, list[Fact]] = {family: [] for family in families}
    for fact in TIER1_FACTS:
        if fact.relation_family in grouped:
            grouped[fact.relation_family].append(fact)
    return grouped


def check_fact_base(facts: tuple[Fact, ...] = TIER1_FACTS) -> list[str]:
    """Integrity of the fact base itself. Returns human-readable violations.

    A duplicated subject inside a family would mean the relation is not functional, which
    silently turns a crossed cell into a correct one; that is the failure this catches.
    """
    violations: list[str] = []

    subject_counts = Counter((f.relation_family, f.subject_slug) for f in facts)
    for (family, subject), count in sorted(subject_counts.items()):
        if count > 1:
            violations.append(
                f"{family}: subject {subject!r} appears {count} times (not functional)"
            )

    for fact in facts:
        if fact.relation_family not in RELATION_FAMILIES:
            violations.append(f"{fact.fact_id}: unknown family {fact.relation_family!r}")
        if not fact.subject_slug:
            violations.append(
                f"{fact.relation_family}: subject {fact.subject!r} slugifies to empty"
            )
        if not fact.answer_slug:
            violations.append(f"{fact.fact_id}: answer {fact.answer!r} slugifies to empty")
        if fact.subject not in fact.subject_phrase:
            violations.append(
                f"{fact.fact_id}: subject_phrase {fact.subject_phrase!r} "
                f"does not contain the subject"
            )
        if fact.answer_slug in fact.also_correct:
            violations.append(f"{fact.fact_id}: also_correct repeats the gold answer")

    by_family: dict[str, set[str]] = {}
    for fact in facts:
        by_family.setdefault(fact.relation_family, set()).add(fact.answer_slug)
    for fact in facts:
        for slug in fact.also_correct:
            if slug not in by_family[fact.relation_family]:
                violations.append(
                    f"{fact.fact_id}: also_correct slug {slug!r} is not an answer in this family"
                )

    return violations
