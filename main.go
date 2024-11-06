package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"sort"

	tf "github.com/wamuir/graft/tensorflow"
	"github.com/wamuir/graft/tensorflow/op"
)

var (
	graphFile = "model/tensorflow_inception_graph.pb"
	labelsFile = "model/imagenet_comp_graph_label_strings.txt"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <image file>")
		return
	}

	imageFile := os.Args[1]
	f, err := os.Open(imageFile)
	if err != nil {
		panic(err)
	}

	modelGraph, labels, err := loadGraphAndLabels()
	if err != nil {
		panic(err)
	}
	defer f.Close()

	session, err := tf.NewSession(modelGraph, nil)
	if err != nil {
		log.Fatalf("Error creating session: %v", err)
	}
	defer session.Close()

	tensor, err := normalizeImage(f)
	if err != nil {
		log.Fatalf("Error normalizing image: %v", err)
	}

	result, err := session.Run(map[tf.Output]*tf.Tensor{
		modelGraph.Operation("input").Output(0): tensor,
	}, []tf.Output{
		modelGraph.Operation("output").Output(0),
	}, nil)
	if err != nil {
		log.Fatalf("Error running session: %v", err)
	}

	probabilities := result[0].Value().([][]float32)[0]

	topFive := getTopFiveLabels(labels, probabilities)

	fmt.Print("\n\n")

	for _, label := range topFive[0:3] {
		fmt.Printf("%s (%2.1f%%)\n", label.Label, label.Probability * 100)
	}

	fmt.Print("\n\n")
}

type Label struct {
	Label string
	Probability float32
}

type Labels []Label

func (l Labels) Len() int { 
	return len(l) 
}
func (l Labels) Less(i, j int) bool {
	return l[i].Probability > l[j].Probability 
}
func (l Labels) Swap(i, j int) {
	l[i], l[j] = l[j], l[i] 
}

func getTopFiveLabels(labels []string, probabilities []float32) []Label {
	var results []Label
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		results = append(results, Label{labels[i], p})
	}

	sort.Sort(Labels(results))
	return results[:5]
}

func normalizeImage(body io.ReadCloser) (*tf.Tensor, error) {
	var buf bytes.Buffer
	io.Copy(&buf, body)

	t, err := tf.NewTensor(buf.String())
	if err != nil {
		return nil, err
	}

	graph, input, output, err := getNormalizedGraph()
	if err != nil {
		return nil, err
	}

	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer sess.Close()

	normalized, err := sess.Run(map[tf.Output]*tf.Tensor{
			input: t,
		},
		[]tf.Output{output}, 
		nil,
	)
	if err != nil {
		return nil, err
	}

	// We only supplied one input, so we only expect one output.
	return normalized[0], nil

}

func getNormalizedGraph() (*tf.Graph, tf.Output, tf.Output, error) {
	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	output := op.Sub(s, 
		op.ResizeBilinear(s,
			op.ExpandDims(s,
				op.Cast(s, decode, tf.Float),
				op.Const(s.SubScope("make_batch"), int32(0)),
			),
			op.Const(s.SubScope("size"), []int32{224, 224}),
		),
		op.Const(s.SubScope("mean"), float32(117)),
		)

	graph, err := s.Finalize()
	if err != nil {
		return nil, tf.Output{}, tf.Output{}, err
	}

	return graph, input, output, nil
			
}

func loadGraphAndLabels() (*tf.Graph, []string, error) {
	model, err := os.ReadFile(graphFile)
	if err != nil {
		return nil, nil, err
	}

	graph := tf.NewGraph()
	if err = graph.Import(model, ""); err != nil {
		return nil, nil, err
	}

	lf, err := os.Open(labelsFile)
	if err != nil {
		return nil, nil, err
	}
	defer lf.Close()

	var labels []string
	scanner := bufio.NewScanner(lf)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}



	return graph, labels, nil
}
