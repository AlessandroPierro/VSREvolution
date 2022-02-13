package it.units.erallab.builder.evolver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.evolver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.representation.sequence.UniformCrossover;
import it.units.malelab.jgea.representation.sequence.bit.BitFlipMutation;
import it.units.malelab.jgea.representation.sequence.bit.BitString;
import it.units.malelab.jgea.representation.sequence.bit.BitStringFactory;

import java.util.Map;

/**
 * @author eric
 */
public class BitsStandard implements EvolverBuilder<BitString> {

  private final int nPop;
  private final int nTournament;
  private final double xOverProb;
  private final boolean diversityEnforcement;
  private final boolean remap;

  public BitsStandard(int nPop, int nTournament, double xOverProb, boolean diversityEnforcement, boolean remap) {
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.xOverProb = xOverProb;
    this.diversityEnforcement = diversityEnforcement;
    this.remap = remap;
  }

  @Override
  public <T, F> Evolver<BitString, T, F> build(PrototypedFunctionBuilder<BitString, T> builder, T target, PartialComparator<F> comparator) {
    int length = builder.exampleFor(target).size();
    BitStringFactory bitsFactory = new BitStringFactory(length);
    double pMut = Math.max(0.01, 1d / (double) length);
    Map<GeneticOperator<BitString>, Double> geneticOperators = Map.of(
        new BitFlipMutation(pMut), 1d - xOverProb,
        new UniformCrossover<>(bitsFactory).andThen(new BitFlipMutation(pMut)), xOverProb
    );
    if (!diversityEnforcement) {
      return new StandardEvolver<>(
          builder.buildFor(target),
          bitsFactory,
          comparator.comparing(Evolver.Individual::fitness),
          nPop,
          geneticOperators,
          new Tournament(nTournament),
          new Last(),
          nPop,
          true,
          remap
      );
    }
    return new StandardWithEnforcedDiversityEvolver<>(
        builder.buildFor(target),
        bitsFactory,
        comparator.comparing(Evolver.Individual::fitness),
        nPop,
        geneticOperators,
        new Tournament(nTournament),
        new Last(),
        nPop,
        true,
        remap,
        100
    );
  }

}