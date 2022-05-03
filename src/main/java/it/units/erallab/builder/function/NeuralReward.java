package it.units.erallab.builder.function;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializableBiFunction;
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
    protected final SerializableBiFunction<Double, Grid<Voxel>, double[]> sensorExtractor;
    protected final int inputDimension;

    public NeuralReward(MultiLayerPerceptron.ActivationFunction activationFunction, SerializableBiFunction<Double, Grid<Voxel>, double[]> sensorExtractor, int inputDimension) {
        this.activationFunction = activationFunction;
        this.sensorExtractor = sensorExtractor;
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
            int nOfInputs = inputDimension;
            int[] innerNeurons = innerNeurons(nOfInputs, innerLayerRatio, nOfInnerLayers).clone();

            @Override
            public Function<List<Double>, SerializableFunction<double[], Double>> buildFor(SerializableFunction<double[], Double> function) {
                return values -> value -> new MultiLayerPerceptron(
                        activationFunction,
                        nOfInputs,
                        innerNeurons,
                        1,
                        values.stream().mapToDouble(d -> d).toArray()
                ).apply(value)[0];
                /*new SerializableFunction<>() {
                    @JsonProperty
                    private final int nOfInputs = inputDimension;
                    @JsonProperty
                    private final int[] innerNeurons = innerNeurons(nOfInputs, innerLayerRatio, nOfInnerLayers).clone();
                    @JsonProperty
                    private final int nOfWeights = MultiLayerPerceptron.countWeights(nOfInputs, innerNeurons, 1);

                    @JsonProperty
                    private final MultiLayerPerceptron mlp = new MultiLayerPerceptron(
                            activationFunction,
                            nOfInputs,
                            innerNeurons,
                            1,
                            values.stream().mapToDouble(d -> d).toArray()
                    );

                    @Override
                    public Double apply(double[] value) {
                        return mlp.apply(value)[0];

                };}*/
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
