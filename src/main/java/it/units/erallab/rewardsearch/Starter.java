/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.erallab.rewardsearch;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.NeuralReward;
import it.units.erallab.builder.solver.SimpleES;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredControlFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.ClusteredObservationFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.RLController;
import it.units.erallab.hmsrobots.core.controllers.rl.RewardFunction;
import it.units.erallab.hmsrobots.core.controllers.rl.continuous.ContinuousRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.TabularSARSALambda;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryInputConverter;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.tasks.rllocomotion.RLEnsembleOutcome;
import it.units.erallab.hmsrobots.tasks.rllocomotion.RLLocomotion;
import it.units.erallab.hmsrobots.util.*;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.TotalOrderQualityBasedProblem;
import it.units.malelab.jgea.core.listener.*;
import it.units.malelab.jgea.core.listener.telegram.TelegramProgressMonitor;
import it.units.malelab.jgea.core.listener.telegram.TelegramUpdater;
import it.units.malelab.jgea.core.solver.Individual;
import it.units.malelab.jgea.core.solver.IterativeSolver;
import it.units.malelab.jgea.core.solver.SolverException;
import it.units.malelab.jgea.core.solver.state.POSetPopulationState;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.function.Function;
import java.util.random.RandomGenerator;

import static it.units.erallab.hmsrobots.behavior.PoseUtils.computeCardinalPoses;
import static it.units.erallab.rewardsearch.NamedFunctions.*;
import static it.units.malelab.jgea.core.listener.NamedFunctions.fitness;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class Starter extends Worker {
    public final static Settings PHYSICS_SETTINGS = new Settings();
    public final static double MAX_LEARNING_TIME = 10d;
    public final static double MAX_EPISODE_TIME = 50d;
    public final static int N_AGENTS = 5;

    public Starter(String[] args) {
        super(args);
    }

    public record Problem(Function<SerializableFunction<double[], Double>, RLEnsembleOutcome> qualityFunction,
                          Comparator<RLEnsembleOutcome> totalOrderComparator) implements TotalOrderQualityBasedProblem<SerializableFunction<double[], Double>, RLEnsembleOutcome> {
    }

    public static Function<SerializableFunction<double[], Double>, RLEnsembleOutcome> buildLocomotionTask(Robot robot, SerializableBiFunction<Double, Grid<Voxel>, double[]> transformation) {
        return r -> new RLLocomotion(MAX_LEARNING_TIME, MAX_EPISODE_TIME, N_AGENTS, robot).apply(new RewardFunction() {
            @JsonProperty
            private SerializableBiFunction<Double, Grid<Voxel>, double[]> sensorExtractor = transformation;
            @Override
            public Double apply(Grid<Voxel> entries) {
                return r.apply(sensorExtractor.apply(0d, entries));
            }
        });
    }

    public static void main(String[] args) {
        new Starter(args);
    }

    @Override
    public void run() {

        final RandomGenerator random = new Random(0);
        String shape = "biped-4x3";
        String sensorConfig = "uniform-a+t+r+vxy-0";
        boolean areaAgent = true;
        boolean touchAgent = false;
        boolean rotationAgent = true;
        boolean areaReward = true;
        boolean touchReward = false;
        boolean rotationReward = false;
        double controllerStep = 0.5;
        int nClusters = 4;


        double episodeTime = d(a("episodeTime", "10"));
        double episodeTransientTime = d(a("episodeTransientTime", "1"));
        double validationEpisodeTime = d(a("validationEpisodeTime", Double.toString(episodeTime)));
        double validationTransientTime = d(a("validationTransientTime", Double.toString(episodeTransientTime)));
        double videoEpisodeTime = d(a("videoEpisodeTime", "10"));
        double videoEpisodeTransientTime = d(a("videoEpisodeTransientTime", "0"));
        int[] seeds = ri(a("seed", "0:1"));
        String experimentName = a("expName", "short");
        List<String> terrainNames = l(a("terrain", "flat"));//"hilly-1-10-rnd"));
        List<String> targetShapeNames = l(a("shape", "biped-4x3"));
        List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform-a-0.01"));
        String lastFileName = a("lastFile", "last");
        String bestFileName = a("bestFile", "best");
        String allFileName = a("allFile", null);
        String finalFileName = a("finalFile", "final");
        String validationFileName = a("validationFile", null);
        boolean deferred = a("deferred", "true").startsWith("t");
        String telegramBotId = a("telegramBotId", "5277744567:AAHnMwkTe67sVvz9aP7S0JKNGJip-ZBVPgs");
        long telegramChatId = Long.parseLong(a("telegramChatId", "1882376186"));
        List<String> serializationFlags = l(a("serialization", "last,final")); //last,best,all,final
        boolean output = a("output", "false").startsWith("t");
        boolean detailedOutput = a("detailedOutput", "false").startsWith("t");
        boolean cacheOutcome = a("cache", "false").startsWith("t");


        Function<RLEnsembleOutcome, Double> fitnessFunction = s -> s.results().stream().map(RLEnsembleOutcome.RLOutcome::validationVelocity).mapToDouble(v -> v).average().orElse(0d);


        List<NamedFunction<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?>> basicFunctions = basicFunctions();
        List<NamedFunction<? super Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?>> basicIndividualFunctions =
                individualFunctions(
                        fitnessFunction);
        List<NamedFunction<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, ?>> populationFunctions =
                populationFunctions(
                        fitnessFunction);

        List<NamedFunction<? super RLEnsembleOutcome, ?>> basicOutcomeFunctions = basicOutcomeFunctions();

        List<ListenerFactory<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Map<String, Object>>> factories =
                new ArrayList<>();
        ProgressMonitor progressMonitor = new ScreenProgressMonitor(System.out);
        //screen listener
        if (bestFileName == null || output) {
            factories.add(new TabularPrinter<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Map<String, Object>>(Misc.concat(List.of(
                    basicFunctions,
                    populationFunctions,
                    best().then(basicIndividualFunctions)
            )), List.of()));
        }
        //file listeners
        if (lastFileName != null) {
            factories.add(new CSVPrinter<>(Misc.concat(List.of(
                    basicFunctions,
                    populationFunctions,
                    best().then(basicIndividualFunctions),
                    basicOutcomeFunctions.stream().map(f -> f.of(fitness()).of(best())).toList(),
                    best().then(serializationFunction(serializationFlags.contains("last")))
            )), keysFunctions(), new File(lastFileName)).onLast());
        }
        if (bestFileName != null) {
            factories.add(new CSVPrinter<>(Misc.concat(List.of(
                    basicFunctions,
                    populationFunctions,
                    best().then(basicIndividualFunctions),
                    basicOutcomeFunctions.stream().map(f -> f.of(fitness()).of(best())).toList(),
                    best().then(serializationFunction(serializationFlags.contains("last")))
            )), keysFunctions(), new File(bestFileName)));
        }
        if (allFileName != null) {
            List<NamedFunction<? super Pair<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>>, ?>> functions = new ArrayList<>();
            functions.addAll(stateExtractor().then(basicFunctions));
            functions.addAll(individualExtractor().then(basicIndividualFunctions));
            functions.addAll(individualExtractor()
                    .then(serializationFunction(serializationFlags.contains("final"))));
            factories.add(new CSVPrinter<>(
                    functions,
                    keysFunctions(),
                    new File(allFileName)
            ).forEach(populationSplitter()));
        }
        if (finalFileName != null) {
            List<NamedFunction<? super Pair<POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Individual<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>>, ?>> functions = new ArrayList<>();
            functions.addAll(stateExtractor().then(basicFunctions));
            functions.addAll(individualExtractor().then(basicIndividualFunctions));
            functions.addAll(individualExtractor()
                    .then(serializationFunction(serializationFlags.contains("final"))));
            factories.add(new CSVPrinter<>(
                    functions,
                    keysFunctions(),
                    new File(finalFileName)
            ).forEach(populationSplitter()).onLast());
        }
        //telegram listener
        if (telegramBotId != null && telegramChatId != 0) {
            factories.add(new TelegramUpdater<>(List.of(
                    fitnessPlot(fitnessFunction)
            ), telegramBotId, telegramChatId));
            progressMonitor = progressMonitor.and(new TelegramProgressMonitor(telegramBotId, telegramChatId));
        }
        ListenerFactory<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, Map<String, Object>> factory = ListenerFactory.all(
                factories);


        //summarize params
        L.info("Shapes: " + shape);
        L.info("Sensor configs: " + sensorConfig);

        Map<String, Object> keys = Map.ofEntries(
                Map.entry("experiment.name", experimentName),
                Map.entry("seed", 0),
                Map.entry("terrain", "flat"),
                Map.entry("shape", shape),
                Map.entry("sensor.config", sensorConfig),
                Map.entry("episode.time", episodeTime),
                Map.entry("episode.transient.time", episodeTransientTime)
        );

        Listener<? super POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>> listener = factory.build(keys);


        // Create the body and the clusters
        Grid<Voxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape(shape));
        Set<Set<Grid.Key>> clustersSet = computeCardinalPoses(Grid.create(body, Objects::nonNull));
        List<List<Grid.Key>> clusters = clustersSet.stream().map(s -> s.stream().toList()).toList();

        ClusteredObservationFunction sensorExtractor = new ClusteredObservationFunction(clusters, areaReward, touchReward, rotationReward);
        ClusteredObservationFunction observationFunction = new ClusteredObservationFunction(clusters, areaAgent, touchAgent, rotationAgent);
        ClusteredControlFunction controlFunction = new ClusteredControlFunction(clusters);
        // Compute dimensions
        int sensorReadingsDimension = observationFunction.getOutputDimension();
        int actionSpaceDimension = (int) Math.pow(2, nClusters);
        int stateSpaceDimension = (int) Math.pow(2, sensorReadingsDimension);

        // Create binary input converter
        DiscreteRL.InputConverter binaryInputConverter = new BinaryInputConverter(sensorReadingsDimension);

        // Create binary output converter
        DiscreteRL.OutputConverter binaryOutputConverter = new BinaryOutputConverter(nClusters, 0.5);

        // Create Tabular Q-Learning agent
        TabularSARSALambda rlAgentDiscrete = new TabularSARSALambda(
                0.99,
                0.75,
                stateSpaceDimension,
                actionSpaceDimension,
                0d,
                0.1,
                0
        );

        // Create continuous agent from discrete one
        ContinuousRL rlAgent = rlAgentDiscrete.with(binaryInputConverter, binaryOutputConverter);

        // Create the RL controller and apply it to the body
        RLController rlController = new RLController(observationFunction, null, rlAgent, controlFunction);
        StepController stepController = new StepController(rlController, controllerStep);
        Robot robot = new Robot(stepController, SerializationUtils.clone(body));

        Map<String, String> params = new HashMap<>();
        params.put("nPop", "12");
        params.put("nEval", "30");
        NeuralReward neuralReward = new NeuralReward(MultiLayerPerceptron.ActivationFunction.TANH, sensorExtractor, sensorExtractor.getOutputDimension());
        PrototypedFunctionBuilder<List<Double>, SerializableFunction<double[], Double>> protFunBuilder = neuralReward.build(params);

        IterativeSolver<? extends POSetPopulationState<?, SerializableFunction<double[], Double>, RLEnsembleOutcome>, TotalOrderQualityBasedProblem<SerializableFunction<double[], Double>, RLEnsembleOutcome>, SerializableFunction<double[], Double>> solver = new SimpleES(0.35, 0.4).build(params).build(protFunBuilder, null);

        Problem problem = new Problem(buildLocomotionTask(robot, sensorExtractor), Comparator.comparing(fitnessFunction).reversed());

        try {
            Collection<SerializableFunction<double[], Double>> solutions = solver.solve(problem, random, executorService, listener);
            progressMonitor.notify((float) 1, "Done");
        } catch (SolverException e) {
            throw new RuntimeException(e);
        }
    }

}
