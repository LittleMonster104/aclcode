"""Caption-aware CIFAR-10 / Flickr dataset loader for Anchor-Bridge training."""
import json, os, random
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Caption templates (Karpathy-style: 3-5 variations per class) ─────────

CAPTION_TEMPLATES = {
      'airplane': [
           'A white airplane flying over the clouds',
           'An airplane in clear blue sky with wings spread wide',
           'Commercial airliner seen from below against white clouds',
       ],
     'automobile': [
           'A red automobile parked on pavement',
           'Automobile with shiny exterior reflecting sunlight',
           'Car shown from side angle on gray road surface',
       ],
        'bird': [
           'Small bird perched on tree branch in natural setting',
           'Colorful bird with bright plumage near water source',
           'Tiny songbird sitting quietly in outdoor environment',
       ],
          'cat': [
           'Domestic cat with soft fur lying peacefully',
           'Cat resting on grass in garden setting',
           'Close-up of a cat looking directly at camera',
       ],
        'deer': [
           'Deer standing in forest clearing with dappled light',
           'Wild deer in natural meadow habitat with trees',
           'Majestic deer in peaceful forest setting at dawn',
       ],
          'dog': [
           'Golden retriever sitting on green grass lawn',
           'Dog with fluffy fur looking forward at camera',
           'Playful dog running through open field area',
       ],
        'frog': [
           'Green frog sitting near still water in pond',
           'Small toad with textured skin on wet ground',
           'Frog resting near lily pads in marsh area',
       ],
       'horse': [
           'Brown horse standing freely in open pasture',
           'Majestic horse with long flowing mane in meadow',
           'Horse grazing peacefully in green field at sunset',
       ],
        'ship': [
           'Large cargo ship sailing across deep blue ocean',
           'White passenger ship moving through calm waters',
           'Navy vessel cutting through choppy sea waves',
       ],
       'truck': [
           'Red delivery truck parked beside loading dock',
           'Pickup truck driving on dusty country road',
           'Large commercial truck on busy highway',
       ],
}

# ── Synthetic shape dataset (original, no classes) ──────────────────────

SYNTH_CAPTIONS = [
     ["close-up red sphere on table", "red ball in sunlight",
      "shiny sphere, bright red"],
    ["blue cube floating", "cube-shaped object, blue",
      "blue 3D block"],
    ["green triangular shape", "triangle with green fill",
      "pointed green object"],
    ["yellow circle on white", "round yellow disc",
      "sun-like yellow circle"],
    ["purple cross symbol", "cruciform purple shape",
      "purple + sign"],
    ["pink star burst", "five-pointed pink star",
      "sparkle, magenta"],
    ["orange hexagonal tile", "hexagon in orange",
      "six-sided orange figure"],
    ["cyan diamond pattern", "rhombus with cyan fill",
      "tilted square in blue-green"],
    ["brown ellipse shape", "oval in brown tones",
      "elongated brown form"],
    ["gray rectangle block", "flat gray rectangular shape",
      "dark gray flat figure"],
]


class CaptionAwareDataset(Dataset):
    """Dataset supporting three data modes:

     - mode="synthetic": PIL-generated shapes (original approach)
     - mode="cifar10":   CIFAR-10 images with class-based captions
     - mode="flickr":    JSON-indexed images from Karpathy-style caption files
    """

      def __init__(self, data_dir='data', mode='synthetic', split='train',
                    max_samples=None, seed=42):
           self.data_dir = Path(data_dir)
         self.mode = mode
          self.split = split
          self.max_samples = max_samples
           self.seed = seed

           self._load_data()

      def _load_data(self):
          """Load data based on mode."""
        random.seed(self.seed)
          np.random.seed(self.seed)

           if self.mode == 'synthetic':
                self._load_synthetic()
            elif self.mode == 'cifar10':
                 self._load_cifar10()
              elif self.mode == 'flickr':
                 self._load_flickr()
             else:
                raise ValueError('Unknown mode: {}'.format(self.mode))

          # Optionally subsample
           if self.max_samples is not None:
               idx = random.sample(range(len(self.data)), min(
                   self.max_samples, len(self.data)))
               idx.sort()
               self.data = [self.data[i] for i in idx]
               print('Subsampled to {} samples'.format(len(self.data)))

      def _load_synthetic(self):
           """Load PIL-generated shape images with caption variations."""
         self.data = []
           self.num_classes = 10
            shapes = ['red_sphere', 'blue_cube', 'green_triangle',
                       'yellow_circle', 'purple_cross', 'pink_star',
                       'orange_hexagon', 'cyan_diamond', 'brown_ellipse',
                      'gray_rectangle']
            colors = [255, 0, 100, 200, 128, 255, 255, 0, 128, 128]

           for i in range(200):       # 20 samples per class
                color = colors[i % len(colors)]
                size = random.randint(24, 48)
                 image = Image.new('RGB', (64, 64), (30, 30, 30))
                 draw = __import__('PIL.ImageDraw').ImageDraw.Draw(image)
                  draw.ellipse([32 - size // 2, 8,
                                   32 + size // 2, 56],
                                  fill=(color, color // 2,
                                         (255 - color) % 256))
                img_path = self.data_dir / 'synthetic' / shapes[i % 10] / f'{i}.png'
                  img_path.parent.mkdir(parents=True, exist_ok=True)
                 image.save(img_path)
                 caption_pool = SYNTH_CAPTIONS[i % 10]
                  caption = random.choice(caption_pool)
                self.data.append({
                      'image_path': str(img_path),
                    'caption': caption,
                     'label': i % 10,
                 })

      def _load_cifar10(self):
           """Load CIFAR-10 images with class-based captions."""
         cifar_dir = self.data_dir / 'cifar-10-batches-bin'
            if not cifar_dir.exists():
               print('[WARNING] CIFAR-10 bin dir not found, using synthetic data')
                 self.mode = 'synthetic'
                  self._load_synthetic()
                  return

          # Read class names
         with open(cifar_dir / 'batches.meta.txt') as f:
                lines = f.read().strip().split('\n')
            class_names = []
              for line in lines:
                   if '=' in line:
                         key, val = line.split('=', 1)
                         if 'label_names' in key:
                               class_names = [x.strip() for x in
                                                  val.strip('[]').replace("'", '').split(',')]

          # Load all batch files
            import struct
           img_dir = self.data_dir / 'cifar10_images'
              img_dir.mkdir(exist_ok=True)

              batch_files = ['data_batch_{}.bin'.format(i) for i in range(1, 6)]
              if self.split == 'test':
                   batch_files = ['test_batch.bin']
                  elif self.split != 'train':
                       batch_files = ['data_batch_{}.bin'.format(i)
                                       for i in (1, 2, 3)]

               self.data = []
              img_size = 32 * 32 * 3
                total = 0

                 for bf in batch_files:
                    filepath = cifar_dir / bf
                       with open(filepath, 'rb') as f:
                              raw = f.read()
                        data = np.frombuffer(raw, dtype=np.uint8)

                         for i in range(0, len(data), img_size + 1):
                               if i + img_size >= len(data):
                                   break
                                  label = int(data[i])
                                  pixels = data[i+1:i+1+img_size] \
                                      .reshape((3, 32, 32)) \
                                       .transpose(1, 2, 0)
                                img_path = img_dir / (bf.replace('.bin', '') +
                                        '_' + str(total).zfill(6) + '.png')
                                  img = Image.fromarray(pixels)
                                   img.save(img_path)
                                  caption_pool = CAPTION_TEMPLATES \
                                    [class_names[label]]
                                   caption = random.choice(caption_pool)
                               self.data.append({
                                     'image_path': str(img_path),
                                       'caption': caption,
                                      'label': label,
                                   })
                                total += 1

                 print('Loaded {} CIFAR-10 images (mode=cifar10, split={})'.format(
                       total, self.split))

      def _load_flickr(self):
          """Load Flickr-style image-caption pairs from JSON index."""
         caption_file = self.data_dir / 'flickr_captions.json'
            if not caption_file.exists():
                print('[WARNING] Flickr captions file not found')
                 self.mode = 'synthetic'
                   self._load_synthetic()
                  return

              flickr_dir = self.data_dir / 'flickr_images'
               flickr_dir.mkdir(exist_ok=True)
               img_dir = flickr_dir / 'images'
               img_dir.mkdir(exist_ok=True)

          all_entries = json.loads(caption_file.read_text())

              if self.split == 'train':
                  idx_range = range(0, int(len(all_entries) * 0.8))
                 elif self.split == 'val':
                   idx_range = range(int(len(all_entries) * 0.8),
                                      int(len(all_entries) * 0.9))
                 else:
                    idx_range = range(int(len(all_entries) * 0.9),
                                       len(all_entries))

              self.data = []
               for i in idx_range:
                   entry = all_entries[i]
                    img_id = str(entry.get('imgid', i))
                       img_path = img_dir / '{}.jpg'.format(img_id)

                        if not img_path.exists():
                            self.data.append({
                                 'image_path': str(img_path),
                                'caption': 'a synthetic image placeholder',
                                  'label': 0,
                               })
                          else:
                               self.data.append({
                                     'image_path': str(img_path),
                                   'caption': random.choice(
                                          entry.get('sentences', [{}])) \
                                        .get('raw', 'a photo'),
                                      'label': 0,
                                  })

                  print('Loaded {} Flickr samples (split={})'.format(
                         len(self.data), self.split))

      def __len__(self):
          return len(self.data)

     def __getitem__(self, idx):
           item = self.data[idx]
             image = Image.open(item['image_path']).convert('RGB')
             caption = item['caption']

            return {
              'image': image,
                'caption': caption,
               'label': item['label'],
                  'img_path': item['image_path'],
           }


if __name__ == '__main__':
     # Quick test
    ds = CaptionAwareDataset('/workspace/data', mode='cifar10',
                                split='train', max_samples=5)
    print('Dataset size: {}'.format(len(ds)))
    sample = ds[0]
    print('Sample keys:', sorted(sample.keys()))
    print('Image shape:', sample['image'].size)
    print('Caption:', sample['caption'][:60])
