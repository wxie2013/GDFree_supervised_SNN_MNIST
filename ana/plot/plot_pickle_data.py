from brian2 import *
import pickle, argparse 


def plot_data(idx, filename):
    with open(filename, 'rb') as f: 
        data = pickle.load(f)
    
    plt.imshow(data['img'][idx], cmap='gray')
    print('label: ', data['label'][idx])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='basic input')
    parser.add_argument('--idx', type=int, required=True, help = 'image index in this data sample')
    parser.add_argument('--file', type=str, required=True, help = 'pickle file name')
    args = parser.parse_args()

    plot_data(args.idx, args.file)
