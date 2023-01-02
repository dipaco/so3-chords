import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from tools.utils.ntu_read_skeleton import read_xyz

NTU_CLASS_NAMES = [
    "A1. drink water",
    "A2. eat meal/snack",
    "A3. brushing teeth",
    "A4. brushing hair",
    "A5. drop",
    "A6. pickup",
    "A7. throw",
    "A8. sitting down",
    "A9. standing up (from sitting position)",
    "A10. clapping",
    "A11. reading",
    "A12. writing",
    "A13. tear up paper",
    "A14. wear jacket",
    "A15. take off jacket",
    "A16. wear a shoe",
    "A17. take off a shoe",
    "A18. wear on glasses",
    "A19. take off glasses",
    "A20. put on a hat/cap",
    "A21. take off a hat/cap",
    "A22. cheer up",
    "A23. hand waving",
    "A24. kicking something",
    "A25. reach into pocket",
    "A26. hopping (one foot jumping)",
    "A27. jump up",
    "A28. make a phone call/answer phone",
    "A29. playing with phone/tablet",
    "A30. typing on a keyboard",
    "A31. pointing to something with finger",
    "A32. taking a selfie",
    "A33. check time (from watch)",
    "A34. rub two hands together",
    "A35. nod head/bow",
    "A36. shake head",
    "A37. wipe face",
    "A38. salute",
    "A39. put the palms together",
    "A40. cross hands in front (say stop)",
    "A41. sneeze/cough",
    "A42. staggering",
    "A43. falling",
    "A44. touch head (headache)",
    "A45. touch chest (stomachache/heart pain)",
    "A46. touch back (backache)",
    "A47. touch neck (neckache)",
    "A48. nausea or vomiting condition",
    "A49. use a fan (with hand or paper)/feeling warm",
    "A50. punching/slapping other person",
    "A51. kicking other person",
    "A52. pushing other person",
    "A53. pat on back of other person",
    "A54. point finger at the other person",
    "A55. hugging other person",
    "A56. giving something to other person",
    "A57. touch other person's pocket",
    "A58. handshaking",
    "A59. walking towards each other",
    "A60. walking apart from each other",
]

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='data/NTU-RGB-D/nturgb+d_skeletons')
    parser.add_argument(
        '--ignored_sample_path',
        default='resource/NTU-RGB-D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/NTU-RGB-D')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
