import os

# English to Khmer animal names dictionary
animal_translation = {
    'antelope': 'សត្វអាន់ទីឡប',
    'badger': 'សត្វបែជឺ',
    'bat': 'សត្វប្រចៀវ',
    'bear': 'ខ្លាឃ្មុំ',
    'bee': 'ឃ្មុំ',
    'beetle': 'សត្វកន្ទុយក',
    'bison': 'សត្វប៊ីសុង',
    'boar': 'សត្វជ្រូកព្រៃ',
    'butterfly': 'មេអំបៅ',
    'cat': 'ឆ្មា',
    'caterpillar': 'សត្វពស់កំប្រុក',
    'chimpanzee': 'សត្វជីមផង់ស៊ី',
    'cockroach': 'សត្វកន្លាត',
    'cow': 'គោ',
    'coyote': 'កូយូទី',
    'crab': 'ក្ដាម',
    'crow': 'សត្វក្អម',
    'deer': 'សត្វក្ដាន់',
    'dog': 'ឆ្កែ',
    'dolphin': 'ដូហ្វីន',
    'donkey': 'លា',
    'dragonfly': 'សត្វចង្រ្កង',
    'duck': 'ទាក្រយ៉ៅ',
    'eagle': 'ឥន្ទ្រីយ៍',
    'elephant': 'ដំរី',
    'flamingo': 'ហ្វ្លេមេនហ្គោ',
    'fly': 'រុយ',
    'fox': 'សត្វតោ',
    'goat': 'ពពែ',
    'goldfish': 'ត្រីមាស',
    'goose': 'ក្ងាន',
    'gorilla': 'សត្វកូរីឡា',
    'grasshopper': 'ចង្រិត',
    'hamster': 'ហាមស្ទ័រ',
    'hare': 'សត្វទន្សាយ',
    'hedgehog': 'សត្វកំប្រុក',
    'hippopotamus': 'សត្វរមៀត',
    'hornbill': 'សត្វក្ងោក',
    'horse': 'សេះ',
    'hummingbird': 'សត្វត្រចៀកកាំ',
    'hyena': 'ហ៊ីណា',
    'jellyfish': 'ពពែកញ្ជ្រែង',
    'kangaroo': 'កង្កែប',
    'koala': 'កូអាឡា',
    'ladybugs': 'សត្វកន្ទុយកមាស',
    'leopard': 'សត្វចាប់',
    'lion': 'សត្វសុត',
    'lizard': 'សត្វកន្លាត',
    'lobster': 'បង្កង',
    'mosquito': 'មូស',
    'moth': 'សត្វម៉ូត',
    'mouse': 'កណ្ដុរ',
    'octopus': 'ប្រទីប',
    'okapi': 'អូកាពី',
    'orangutan': 'អូរេងអូទាន',
    'otter': 'ខ្លាចាស់',
    'owl': 'សត្វបក្សីព្រិល',
    'ox': 'គោពពែ',
    'oyster': 'ប្រហិត',
    'panda': 'បេនឌា',
    'parrot': 'សត្វបាក់តុក',
    'pelecaniformes': 'ពេលីកានីហ្វម',
    'penguin': 'មច្ឆាក្រៀម',
    'pig': 'ជ្រូក',
    'pigeon': 'ព្រាប',
    'porcupine': 'កណ្ដុរញី',
    'possum': 'ប៉ូស៊ុម',
    'raccoon': 'សត្វរ៉ាកូន',
    'rat': 'កណ្ដុរ',
    'reindeer': 'សត្វក្របី',
    'rhinoceros': 'សត្វក្របី',
    'sandpiper': 'សត្វស្ទឹង',
    'seahorse': 'សត្វសេះទឹក',
    'seal': 'សេល',
    'shark': 'ឆ្កែសមុទ្រ',
    'sheep': 'ចៀម',
    'snake': 'ពស់',
    'sparrow': 'សត្វចាប',
    'squid': 'មឹក',
    'squirrel': 'កណ្ដុរត្នោត',
    'starfish': 'ត្រីផ្កាយ',
    'swan': 'សត្វអង្គារ',
    'tiger': 'ខ្លាឃ្មុំ',
    'turkey': 'សត្វតួកគី',
    'turtle': 'អណ្ដើក',
    'whale': 'ត្រីបាឡែន',
    'wolf': 'ចចក',
    'wombat': 'វូមបេត',
    'woodpecker': 'បក្សីកូនឈើ',
    'zebra': 'សត្វរេប្រ៉ា'
}


# Base directory and output file path
base_dir = '/home/vitoupro/code/image_captioning/data/raw/animals'
annotations_file_path = '/home/vitoupro/code/image_captioning/data/raw/annotation.txt'

# Create the annotations file
with open(annotations_file_path, 'w') as file:
    for animal in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, animal)):
            # Check if the animal name has a translation in the dictionary
            khmer_name = animal_translation.get(animal)
            if khmer_name:  # Only proceed if a translation is available
                for image in os.listdir(os.path.join(base_dir, animal)):
                    record = f"animals/{animal}/{image} {khmer_name}\n"
                    file.write(record)

print(f"Annotation file created at {annotations_file_path}")
