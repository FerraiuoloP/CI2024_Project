<!-- omit in toc -->
# Symbolic Regression using Genetic Programming

This repository contains an implementation of a Symbolic Regression algorithm, using a tree-based **Genetic Programming** (GP) evolutionary technique. The algorithm evolves mathematical expressions in order to find the model that best fits a given set of data in the form $(X, y)$. By leveraging *selection*, according to a fitness measure, *mutation* and *crossover*, the SR algorithm generates mathematical formulas that are able to capture the complex patterns present in the input data.

## Key Features
- **Island Model Genetic Algorithm**
  - The population is divided into a certain number of subpopulations (a.k.a. *islands*) which evolve separately and can occasionaly exchange individuals (a.k.a *migration*) in order to avoid local optima and rapid convergence;
  - At each migration event, according to a migration rate parameter, one or more individuals migrate from the source island to another random island, as a way to ensure equal chance of genetic mixing across islands.
- **Tree-Based Representation**
  - The evolutionary algorithm iteratively evolves a population of mathematical formulas, represented as full and grow trees. Internal nodes are randomly chosen from function set (operators), while leaves are randomly chosen from terminal set (constants and variables).
- **Elitism**
  - To preserve high-quality solutions, the best individuals (a.k.a. *elites*) are directly inserted into the next generation, without being subjected to any change.
- **Parents Selection**
  - Different parents selection strategies are implemented. Fitness-proportional, rank based and tournament selection.
- **Mutation and Crossover**
  - Various mutation mechanisms are implemented. Replaced a subtree with a new one (`mutate_subtree`) or modify a single node (`mutate_single_node`) in the selected parent tree;
  - Combine two different trees for generating new offsprings. This allows the algorithm to explore new regions in the search space, encouraging exploration instead of exploitation.

## How it works
- **Initialization**
  - A population of individuals (*trees*) is initialized on each island. Depending on the value assigned to the variable `grow_full_ratio`, each island's population is initialized with a certain number of full trees and grow trees;
  - In each of the `ISLAND_NUM` islands, there are `ISLAND_POPULATION` individuals.
- **Selection**
  - Parents are selected based on their fitness (in which measure the mathematical formula represented by the tree fits well the data provided as input), using various strategies (*e.g.* rank-based selection).
- **Reproduction**
  - Offsprings are generated through mutation and crossover genetic operators.
- **Evolution**
  - Over the course of generations, populations on the islands evolve and only the best performing trees survive.
- **Convergence**
  - The evolutionary process continues for a certain number of generations (`MAX_GENERATIONS`).

## Contributing
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/AgneseRe">
        <img src="https://github.com/AgneseRe.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>AgneseRe</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/FerraiuoloP">
        <img src="https://github.com/FerraiuoloP.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>FerraP</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/GDennis01">
        <img src="https://github.com/GDennis01.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>GDennis01</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/XhoanaShkajoti">
        <img src="https://github.com/XhoanaShkajoti.png" width="50px" style="border-radius: 50%; border: none;" alt=""/>
        <br />
        <sub>XhoanaShkajoti</sub>
      </a>
    </td>
  </tr>
</table>

## License