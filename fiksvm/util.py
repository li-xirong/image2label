
import time

def classifyLargeData(model, imset, feat_file, prob_output=False, blocksize=1000):
    start = 0
    results = []

    read_time = 0.0
    test_time = 0.0

    while start < len(imset):
        end = min(len(imset), start + blocksize)
        #printStatus(INFO, 'classifyLargeData: classifying images from %d to %d' % (start, end-1))

        s_time = time.time()
        renamed,vectors = feat_file.read(imset[start:end])
        read_time += time.time() - s_time
    
        s_time = time.time()
        if prob_output:
            scores = [model.predict_probability(vectors[i]) for i in range(len(renamed))]
        else:
            scores = [model.predict(vectors[i]) for i in range(len(renamed))]
        test_time += time.time() - s_time

        results += zip(renamed, scores)
        start = end

    print ('[classifyLargeData] read time %g seconds, test time %g seconds' % (read_time, test_time))
    results.sort(key=lambda v: (v[1], v[0]), reverse=True)
    return results


    