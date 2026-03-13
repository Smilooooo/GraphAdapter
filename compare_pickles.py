import pickle
import os

ga = r'E:\BachelorThesis\Data\data(3)\data\LungHist700\split_fewshot_ga\shot_4-seed_11111.pkl'
rb = r'E:\BachelorThesis\Data\data(3)\data\LungHist700\split_fewshot_rebased\shot_4-seed_11111.pkl'

with open(ga, 'rb') as f:
    d_ga = pickle.load(f)
with open(rb, 'rb') as f:
    d_rb = pickle.load(f)

print('=== GA train (first 5) ===')
for d in d_ga['train'][:5]:
    print(f'  label={d.label}, class={d.classname}, path={d.impath}')

print('\n=== RB train (first 5) ===')
for d in d_rb['train'][:5]:
    print(f'  label={d.label}, class={d.classname}, path={d.impath}')

ga_train = d_ga['train']
rb_train = d_rb['train']
ga_val = d_ga['val']
rb_val = d_rb['val']

print(f'\nGA train={len(ga_train)}, val={len(ga_val)}')
print(f'RB train={len(rb_train)}, val={len(rb_val)}')

# Compare filenames
ga_files = sorted([os.path.basename(d.impath) for d in ga_train])
rb_files = sorted([os.path.basename(d.impath) for d in rb_train])
print(f'\nSame train filenames (sorted): {ga_files == rb_files}')

ga_labels = [d.label for d in ga_train]
rb_labels = [d.label for d in rb_train]
print(f'Same train labels (order): {ga_labels == rb_labels}')

ga_classes = [d.classname for d in ga_train]
rb_classes = [d.classname for d in rb_train]
print(f'Same train classnames (order): {ga_classes == rb_classes}')

# Show path differences
print('\n=== Train path diffs ===')
diff_count = 0
for i, (g, r) in enumerate(zip(ga_train, rb_train)):
    if g.impath != r.impath:
        diff_count += 1
        if diff_count <= 5:
            print(f'  [{i}] GA: {g.impath}')
            print(f'       RB: {r.impath}')
print(f'Total path diffs: {diff_count} / {len(ga_train)}')

# Val comparison
ga_vfiles = sorted([os.path.basename(d.impath) for d in ga_val])
rb_vfiles = sorted([os.path.basename(d.impath) for d in rb_val])
print(f'\nSame val filenames (sorted): {ga_vfiles == rb_vfiles}')
