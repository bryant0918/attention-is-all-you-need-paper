import pandas as pd

# Create DataFrame
dataset1 = {
    'German': ['Guten Morgen!', 'Wie geht es dir?', 'Ich bin hungrig.', 'Entschuldigung, wo ist die Toilette?', 'Wie viel kostet das?',
               'Ich spreche kein Deutsch.', 'Was ist dein Name?', 'Es tut mir leid.', 'Woher kommst du?', 'Ich liebe dich.',
               'Wie spät ist es?', 'Kannst du mir helfen?', 'Ich verstehe nicht.', 'Auf Wiedersehen!', 'Wo ist der Bahnhof?',
               'Ich habe eine Frage.', 'Wie alt bist du?', 'Ich bin müde.', 'Was machst du gerne in deiner Freizeit?', 'Was ist das?',
               'Mein Name ist John.', 'Wie heißt das auf Deutsch/Englisch?', 'Ich bin beschäftigt.', 'Wie war dein Tag?',
               'Ich habe Hunger.'],
    'English': ['Good morning!', 'How are you?', 'I am hungry.', 'Excuse me, where is the restroom?', 'How much does that cost?',
                "I don't speak German.", 'What is your name?', "I'm sorry.", 'Where are you from?', 'I love you.',
                'What time is it?', 'Can you help me?', "I don't understand.", 'Goodbye!', 'Where is the train station?',
                'I have a question.', 'How old are you?', 'I am tired.', 'What do you like to do in your free time?', 'What is that?',
                'My name is John.', 'What is that called in German/English?', 'I am busy.', 'How was your day?', 'I am hungry.']
}

df1 = pd.DataFrame(dataset1)

# print(df)




# German sentences
german_sentences = [
    "Die Sonne scheint am blauen Himmel über den Bergen.",
    "Ich gehe mit meinem Hund im Park spazieren.",
    "Wir essen gerne Pizza und trinken kühles Bier.",
    "Meine Schwester liest ein Buch in ihrem Zimmer.",
    "Der Zug fährt pünktlich zum Bahnhof und bringt die Passagiere.",
    "Im Garten blühen bunte Blumen und grüne Sträucher.",
    "Er spielt Gitarre und singt schöne Lieder auf der Bühne.",
    "Die Kinder spielen fröhlich im Park und lachen laut.",
    "Der Kellner serviert köstliches Essen und erfrischende Getränke.",
    "Ich höre gerne klassische Musik und entspanne mich.",
    "Die Katze schläft gemütlich auf dem weichen Sofa im Wohnzimmer.",
    "Am Strand spazieren gehen und das Rauschen des Meeres hören.",
    "Der Flugzeug fliegt hoch in den Himmel und lässt Wolken zurück.",
    "In der Küche kochen wir zusammen leckeres Essen für die Familie.",
    "Die Vögel singen früh am Morgen ein schönes Lied.",
    "Die Blätter fallen von den Bäumen im Herbstwind.",
    "Das Auto fährt schnell auf der Autobahn Richtung Norden.",
    "Die Schüler lernen fleißig in der Schule und machen ihre Hausaufgaben.",
    "Wir besuchen unsere Großeltern und verbringen Zeit zusammen.",
    "Der Regen prasselt gegen das Fenster und lässt die Blumen im Garten wachsen.",
    "Der Hund bellt laut, wenn er einen Fremden sieht.",
    "Die Stadt ist groß und voller Menschen, die einkaufen und arbeiten.",
    "Wir fahren in den Urlaub und besuchen schöne Orte am Meer.",
    "Die Sterne leuchten hell am dunklen Nachthimmel.",
    "Die Uhr tickt langsam und zeigt die Zeit an.",
    "Die Eule fliegt leise durch die Nacht und sucht nach Beute.",
    "Der Lehrer erklärt das Thema und beantwortet Fragen der Schüler.",
    "Die Menschen tanzen fröhlich bei der Party und genießen die Musik.",
    "Das Schiff segelt über das Meer und erreicht bald den Hafen.",
    "Die Wolken ziehen am Himmel vorbei und bringen Regen.",
    "Die Bienen fliegen von Blume zu Blume und sammeln Nektar.",
    "Die Kinder spielen im Park und toben herum.",
    "Der Wald ist dunkel und geheimnisvoll, aber auch schön.",
    "Die Katze schleicht sich leise an ihre Beute heran.",
    "Der Himmel ist blau und die Sonne scheint warm.",
    "Die Berge sind hoch und majestätisch, und die Aussicht ist atemberaubend.",
    "Die Menschen gehen spazieren und genießen die frische Luft.",
    "Die Blumen blühen im Frühling und verbreiten ihren Duft.",
    "Die Kinder spielen im Garten und haben Spaß.",
    "Die Vögel zwitschern fröhlich am Morgen und begrüßen den neuen Tag.",
    "Die Schafe grasen friedlich auf der Weide und genießen die Sonne.",
    "Die Straßen sind belebt und voller Menschen, die unterwegs sind.",
    "Die Wellen brechen am Ufer und bringen Muscheln und Seetang.",
    "Die Luft ist frisch und klar, und die Natur ist schön.",
    "Die Sonne geht langsam unter und taucht den Himmel in rotes Licht.",
    "Die Stadt ist voller Leben und bietet viele Möglichkeiten.",
    "Die Hunde rennen fröhlich über die Wiese und spielen miteinander.",
    "Die Fische schwimmen im klaren Wasser des Sees und sind glücklich.",
    "Die Sterne funkeln am Nachthimmel und machen die Nacht schön.",
    "Die Blätter rascheln im Wind und fallen langsam."
]

# English translations
english_translations = [
    "The sun shines in the blue sky over the mountains.",
    "I walk with my dog in the park.",
    "We enjoy eating pizza and drinking cold beer.",
    "My sister is reading a book in her room.",
    "The train departs on time to the station and brings the passengers.",
    "In the garden, colorful flowers and green bushes bloom.",
    "He plays the guitar and sings beautiful songs on stage.",
    "The children play happily in the park and laugh loudly.",
    "The waiter serves delicious food and refreshing drinks.",
    "I enjoy listening to classical music and relaxing.",
    "The cat sleeps comfortably on the soft sofa in the living room.",
    "Take a walk on the beach and listen to the sound of the sea.",
    "The airplane flies high into the sky and leaves clouds behind.",
    "In the kitchen, we cook delicious food together for the family.",
    "The birds sing a beautiful song early in the morning.",
    "The leaves fall from the trees in the autumn wind.",
    "The car drives quickly on the highway towards the north.",
    "The students study diligently at school and do their homework.",
    "We visit our grandparents and spend time together.",
    "The rain beats against the window and makes the flowers in the garden grow.",
    "The dog barks loudly when he sees a stranger.",
    "The city is big and full of people who shop and work.",
    "We go on vacation and visit beautiful places by the sea.",
    "The stars shine brightly in the dark night sky.",
    "The clock ticks slowly and shows the time.",
    "The owl flies silently through the night and searches for prey.",
    "The teacher explains the topic and answers the students' questions.",
    "People dance happily at the party and enjoy the music.",
    "The ship sails across the sea and soon reaches the harbor.",
    "The clouds drift across the sky and bring rain.",
    "The bees fly from flower to flower and collect nectar.",
    "The children play in the park and romp around.",
    "The forest is dark and mysterious, but also beautiful.",
    "The cat creeps silently towards its prey.",
    "The sky is blue and the sun shines warmly.",
    "The mountains are high and majestic, and the view is breathtaking.",
    "People go for walks and enjoy the fresh air.",
    "The flowers bloom in spring and spread their fragrance.",
    "The children play in the garden and have fun.",
    "The birds chirp happily in the morning and greet the new day.",
    "The sheep graze peacefully in the pasture and enjoy the sun.",
    "The streets are bustling and full of people who are on the go.",
    "The waves break on the shore and bring shells and seaweed.",
    "The air is fresh and clear, and nature is beautiful.",
    "The sun slowly sets and bathes the sky in red light.",
    "The city is full of life and offers many opportunities.",
    "The dogs run happily across the meadow and play together.",
    "The fish swim in the clear water of the lake and are happy.",
    "The stars twinkle in the night sky and make the night beautiful.",
    "The leaves rustle in the wind and fall slowly."
]

# Create DataFrame
df2 = pd.DataFrame({
    'German': german_sentences,
    'English': english_translations
})

# print(df)




