# Copyright (c) 2020 Mobvoi Inc (Di Wu)
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#                    Jing Du
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import glob

import yaml
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--is_hifigan',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    val_scores = []
    if args.val_best:
        yamls = glob.glob('{}/*.yaml'.format(args.src_path))
        yamls = [
            f for f in yamls
            if not (os.path.basename(f).startswith('train')
                    or os.path.basename(f).startswith('init'))
        ]
        for y in yamls:
            with open(y, 'r') as f:
                dic_yaml = yaml.load(f, Loader=yaml.BaseLoader)
                loss = float(dic_yaml['loss_dict']['loss'])
                epoch = int(dic_yaml['epoch'])
                step = int(dic_yaml['step'])
                tag = dic_yaml['tag']
                name = os.path.basename(y)
                val_scores += [[epoch, step, loss, tag, name]]
        sorted_val_scores = sorted(val_scores,
                                   key=lambda x: x[2],
                                   reverse=False)
        print("best val (epoch, step, loss, tag) = " +
              str(sorted_val_scores[:args.num]))
        path_list = [
            args.src_path + f"/{score[-1].replace('yaml', 'pt')}"
            for score in sorted_val_scores[:args.num]
        ]
    print(path_list)
    avg = {}
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        for k in states.keys():
            k1 = k
            if args.is_hifigan:
                if not k.startswith('generator'):
                    continue
                else:
                    k1 = k.replace("generator.", "")

            if k1 not in avg.keys():
                avg[k1] = states[k].clone()
            else:
                avg[k1] += states[k]
    # average
    new_dict = {}
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)

            if k.startswith('generator') and args.is_hifigan:
                new_k = k.replace("generator.", '')
                new_dict[new_k] = avg[k]
            else:
                new_dict[k] = avg[k]

    print('Saving to {}'.format(args.dst_model))
    torch.save(new_dict, args.dst_model)


if __name__ == '__main__':
    main()
