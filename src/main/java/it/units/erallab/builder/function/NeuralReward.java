package it.units.erallab.builder.function;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.util.SerializableFunction;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * @author eric
 */
public class NeuralReward implements NamedProvider<PrototypedFunctionBuilder<List<Double>, SerializableFunction<double[], Double>>> {

    protected final MultiLayerPerceptron.ActivationFunction activationFunction;
    protected final int inputDimension;

    public NeuralReward(MultiLayerPerceptron.ActivationFunction activationFunction, int inputDimension) {
        this.activationFunction = activationFunction;
        this.inputDimension = inputDimension;
    }

    protected static int[] innerNeurons(int nOfInputs, double innerLayerRatio, int nOfInnerLayers) {
        int[] innerNeurons = new int[nOfInnerLayers];
        int centerSize = (int) Math.max(2, Math.round(nOfInputs * innerLayerRatio));
        if (nOfInnerLayers > 1) {
            for (int i = 0; i < nOfInnerLayers / 2; i++) {
                innerNeurons[i] = nOfInputs + (centerSize - nOfInputs) / (nOfInnerLayers / 2 + 1) * (i + 1);
            }
            for (int i = nOfInnerLayers / 2; i < nOfInnerLayers; i++) {
                innerNeurons[i] = centerSize + (1 - centerSize) / (nOfInnerLayers / 2 + 1) * (i - nOfInnerLayers / 2);
            }
        } else if (nOfInnerLayers > 0) {
            innerNeurons[0] = centerSize;
        }
        return innerNeurons;
    }

    @Override
    public PrototypedFunctionBuilder<List<Double>, SerializableFunction<double[], Double>> build(Map<String, String> params) {
        double innerLayerRatio = Double.parseDouble(params.getOrDefault("r", "0.65"));
        int nOfInnerLayers = Integer.parseInt(params.getOrDefault("nIL", "1"));
        return new PrototypedFunctionBuilder<>() {

            @Override
            public Function<List<Double>, SerializableFunction<double[], Double>> buildFor(SerializableFunction<double[], Double> function) {
                return values -> {
                    MultiLayerPerceptron multiLayerPerceptron = new MultiLayerPerceptron(
                            activationFunction,
                            inputDimension,
                            innerNeurons(inputDimension, innerLayerRatio, nOfInnerLayers).clone(),
                            1,
                            values.stream().mapToDouble(d -> d).toArray()
                    );
                    return value -> multiLayerPerceptron.apply(value)[0];
                };
            }

            @Override
            public List<Double> exampleFor(SerializableFunction<double[], Double> function) {
                return Collections.nCopies(
                        MultiLayerPerceptron.countWeights(
                                MultiLayerPerceptron.countNeurons(
                                        inputDimension,
                                        innerNeurons(
                                                inputDimension,
                                                innerLayerRatio,
                                                nOfInnerLayers
                                        ),
                                        1
                                )
                        ),
                        0d
                );
            }
        };
    }


}
