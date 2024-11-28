# Genetic Algorithm for GDL Evolution

## Overview
This repository contains a Rust-based implementation of a genetic algorithm designed to evolve Game Description Language (GDL) programs. The evolution is driven by crossover operations focusing on `RuleClause` elements within the GDL, thereby creating variations while maintaining valid syntactic structures.

The primary goal is to generate diverse game rules through genetic crossover techniques and maintain a population of GDL programs that iteratively evolve over generations.

## Features
- **Crossover on Rule Clauses**: The crossover algorithm focuses only on `RuleClause` elements in the GDL programs.
- **Genetic Algorithm**: Uses a population-based approach to iteratively evolve GDL programs.
- **Detailed Statistics**: Tracks and reports crossover operations performed, including the number of subtrees replaced and average subtree sizes.

## Prerequisites
- Rust 1.50+ installed
- Cargo for building and running the program

## Installation
Clone the repository using:
```bash
$ git clone https://github.com/yourusername/gdl_evolution.git
$ cd gdl_evolution
```

Build the project with:
```bash
$ cargo build 
```

## Usage

1. Place the initial GDL files in a directory named `gdl_games`.
2. Run the genetic algorithm using the command:

```bash
$ cargo run
```

This will read the GDL files from the `gdl_games` directory, create an initial population, and evolve them over several generations.

3. The evolved GDL programs will be saved in the `evolved_programs` directory.

## Configuration
The program allows you to configure several parameters:
- **Number of Programs**: Modify `num_programs` in `main()` to change the initial population size.
- **Crossover Rate**: Change `crossover_rate` to control the frequency of crossovers.
- **Generations**: Adjust the `generations` parameter in `genetic_algorithm()` to change how many generations are used.

## Example
1. Create a directory called `gdl_games` and add GDL programs to this directory.
2. Run the program to generate evolved programs.
3. The output files, including differences between parents and evolved programs, will be saved in the `evolved_programs` directory.

## Structure
- **src/**: Source code for the genetic algorithm and its utilities.
- **gdl_games/**: Initial GDL programs directory (needs to be created and populated by the user).
- **evolved_programs/**: Directory where evolved GDL programs are saved.
