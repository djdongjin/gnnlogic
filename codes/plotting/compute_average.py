from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model2name = {
    'thesis_gin': 'GIN',
    'thesis_bilstm': 'biLSTM',
    'thesis_mac': 'MAC',
    'thesis_bilstm_atten': 'biLSTM-attn',
    'thesis_kd_mac_0.5_0.5': 'MAC-0.5-0.5-0',
    'thesis_kd_mac_0.1_0.9': 'MAC-0.1-0.9-0',
    'thesis_kd_mac_0.0_1.0': 'MAC-0.0-1.0-0',
    'thesis_kdmax_batch_mac_0.0_0.0_1': 'MAC-0.0-0.0-1',
    'thesis_kdmax_batch_mac_0.0_1_1': 'MAC-0.0-1-1',
    'thesis_kdmax_batch_mac_0.1_0.6_0.3': 'MAC-0.1-0.6-0.3',
    'thesis_kdmax_batch_mac_0.2_0.4_0.4': 'MAC-0.2-0.4-0.4',
    'thesis_kdmax_batch_mac_0.5_1_1': 'MAC-0.5-1-1',
    'thesis_kdmax_batch_lstm_0.1_0.6_0.3': 'biLSTM-0.1-0.6-0.3',
    'thesis_kdmax_batch_mac_1_1_5': 'MAC-1-1-5',
    'thesis_kdmax_nonbatch_mac_0.5_1_1_test_edge': 'MAC-edge-0.5-1-1',
    'thesis_kdmax_nonbatch_mac_0.5_1_1_test_node': 'MAC-node-0.5-1-1',
    'thesis_kdmax_batch_lstm_attn_0.5_1.0_2.0': 'LSTM-attn-0.5-1-2'
}

output_key_order = 'biLSTM biLSTM-attn MAC GIN MAC-0.5-0.5-0 MAC-0.1-0.9-0 MAC-0.0-1.0-0 MAC-0.0-0.0-1 MAC-0.0-1-1 MAC-0.1-0.6-0.3 MAC-0.2-0.4-0.4 MAC-0.5-1-1'.split()

length_file_list = '''data_1.2,1.3.csv
data_1.3,1.4.csv
data_1.4,1.5.csv
data_1.5,1.6.csv
data_1.6,1.7.csv
data_1.7,1.8.csv
data_1.8,1.9.csv
data_1.9,1.10.csv'''.split('\n')

noise_file_list = '''data_1.2,1.3_clean.csv
data_2.2,2.3_supporting.csv
data_3.2,3.3_irrelevant.csv
data_4.2,4.3_disconnected.csv'''.split('\n')

file_list = length_file_list + noise_file_list

colors = sns.color_palette('colorblind', n_colors=len(output_key_order))


def key2label(k):
    """
	k has a form of length.distractor
	"""
    distractor, l = k.split('.')
    distractor2name = ['Clean', 'Supporting', 'Irrelevant', 'Disconnect']
    distractor = distractor2name[int(distractor) - 1]

    k = '{}\nLength:{}'.format(distractor, l)
    return k


def combine_data(file, write_average=True):
    # all_data: {model_name: [[row1], [row2], ...]}
    all_data = defaultdict(list)

    with open(file) as f:
        lines = [line[:-1].split('&') for line in f.readlines()]
        head_keys = lines[0][3:]
        values = lines[1:]

        for line in values:
            model, val = line[0], list(map(lambda x: float(x) / 100, line[3:]))
            all_data[model2name[model]].append(val)

        if file.count('_') < 2:
            # file corresponds to length experiments, only kept length, as there is no distractors.
            head_keys = [k.split('_')[0].split('.')[1] for k in head_keys]
        else:
            # file corresponds to distractor experiments.
            head_keys = [k.split('_')[0] for k in head_keys]

    all_data = dict(all_data)
    means, stds = {}, {}
    for key in all_data:
        all_data[key] = np.array(all_data[key])
        means[key] = all_data[key].mean(axis=0)
        stds[key] = all_data[key].std(axis=0)

    if write_average:
        with open('mean_' + file, 'w') as writer:
            head_line = ['model_name', ] + head_keys
            head_line = ','.join(head_line)
            writer.write(head_line + '\n')

            for k in all_data.keys():
                all_data[k] = ','.join(['{:.2f}~{:.2f}'.format(x, y) for x, y in zip(means[k], stds[k])])
                # all_data[k] = ','.join(['{:.2f}'.format(x) for x in means[k]])
                writer.write(k + ',' + all_data[k] + '\n')

    return all_data, means, stds, head_keys


def combine_files(files, new_file):
    with open(new_file, 'w') as writer:
        for file in files:
            with open(file) as reader:
                lines = reader.readlines()
                content = ''.join(lines)
                file = file.replace(',', '_').replace('.csv', '')
                content = file + '\n' + content

                writer.write(content + '\n\n')


def draw_axes_length_1(ax, mean, std, idx, name, model_to_include=None):
    for idx_model, k in enumerate(output_key_order):
        if k in mean and (model_to_include is None or k in model_to_include):
            data = sorted([(ii, mm, ss) for (ii, mm, ss) in zip(idx, mean[k], std[k])])
            new_idx, m, s = zip(*data)
            new_idx, m, s = np.array(new_idx), np.array(m), np.array(s)
            ax.plot(new_idx, m, label=k, linewidth=1.5, color=colors[idx_model])
            ax.fill_between(new_idx, m + s, m - s, alpha=0.05, edgecolor=colors[idx_model], facecolor=colors[idx_model])
    ax.legend(loc=1)
    ax.set_xlabel('Relation Length(Test)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Relation Length(Train):' + name)


def draw_figures_length_1(files, model_to_include=None, save_name='length_1.jpg'):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    for ax, f in zip(axes.flat, files[:4]):
        data, mean, std, head_keys = combine_data(f, write_average=False)
        head_keys = list(map(int, head_keys))
        name = f.replace('.csv', '').replace('data_', '').split(',')  # e.g.: data_1.7,1.8.csv -> 1.7 1.8
        name = [n.split('.')[1] for n in name]
        name = ','.join(name)  # -> 7,8
        draw_axes_length_1(ax, mean, std, head_keys, name, model_to_include)

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    for ax, f in zip(axes.flat, files[4:]):
        data, mean, std, head_keys = combine_data(f, write_average=False)
        head_keys = list(map(int, head_keys))
        name = f.replace('.csv', '').replace('data_', '').split(',')  # e.g.: data_1.7,1.8.csv -> 1.7 1.8
        name = [n.split('.')[1] for n in name]
        name = ','.join(name)  # -> 7,8
        draw_axes_length_1(ax, mean, std, head_keys, name, model_to_include)

    plt.savefig('2' + save_name, bbox_inches='tight', pad_inches=0)
    plt.show()


def draw_axes_distractor_1(ax, mean, std, idx, name, model_to_include=None):
    bar_width = 0.2
    bar_n = 4

    delete_idx = 0
    for i in idx:
        if i.split('.')[1] == '2':
            break
        delete_idx += 1
    idx = idx[:delete_idx] + idx[delete_idx + 1:]
    x = np.arange(len(idx))

    for idx_model, k in enumerate(output_key_order):
        if k in mean and (model_to_include is None or k in model_to_include):
            m, s = list(mean[k]), list(std[k])
            m = np.array(m[:delete_idx] + m[delete_idx + 1:])
            s = np.array(s[:delete_idx] + s[delete_idx + 1:])
            idx_offset = -1 if idx_model < bar_n // 2 else 1
            ax.bar(x + int(idx_model + 0.5 - bar_n // 2) * bar_width + idx_offset * (bar_width / 2),
                   m, bar_width, yerr=s, label=k, color=colors[idx_model])
    ax.legend(loc=1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Noise Type(Train):' + name)
    ax.set_xticks(x)
    ax.set_xticklabels(list(map(key2label, idx)))


def draw_figures_distractor_1(files, model_to_include=None, save_name='distractor_1.jpg'):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    for ax, f in zip(axes.flat, files):
        data, mean, std, head_keys = combine_data(f, write_average=False)
        name = f.split('_')[2].replace('.csv', '')  # e.g.: data_1.2,1.3_clean.csv
        draw_axes_distractor_1(ax, mean, std, head_keys, name, model_to_include)

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':

    file2data, file2means, file2stds, file2head_keys = {}, {}, {}, {}
    for f in file_list:
        file2data[f], file2means[f], file2stds[f], file2head_keys[f] = combine_data(f)
    mean_file_list = ['mean_' + filename for filename in file_list]
    combine_files(mean_file_list, 'mean_all.csv')

    # figures for experiments in part 1: single model
    model_to_include_1 = set('biLSTM biLSTM-attn MAC GIN'.split(' '))
    draw_figures_length_1(length_file_list, model_to_include_1)
    draw_figures_distractor_1(noise_file_list, model_to_include_1)

    # figures for experiments in part 2: integration
    model_to_include_2 = [md for md in output_key_order if md != 'GIN']
    draw_figures_length_1(length_file_list, model_to_include_2, save_name='length_2.jpg')
    draw_figures_distractor_1(noise_file_list, model_to_include_2, save_name='distractor_2.jpg')
