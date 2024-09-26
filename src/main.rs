use gdl_parser::{parse, Clause, Description, Relation, Sentence, Term};
use rand::prelude::*;
use std::fs;
use std::path::Path;

fn parse_gdl(program_text: &str) -> Option<Description> {
    // Since `parse` returns a `Description`, you can directly wrap it in `Some`.
    Some(parse(program_text))
}




fn tree_to_gdl(description: &Description) -> String {
    description.to_string()
}

fn find_compatible_subtrees<'a>(
    clause: &'a Clause,
    target_relation: &Relation,
) -> Vec<&'a Relation> {
    let mut compatible_subtrees = Vec::new();

    if let Clause::SentenceClause(sentence) = clause {
        if let Sentence::RelSentence(relation) = sentence {
            if are_relations_compatible(relation, target_relation) {
                compatible_subtrees.push(relation);
            }
        }
    }

    compatible_subtrees
}

fn are_relations_compatible(rel1: &Relation, rel2: &Relation) -> bool {
    // Two relations are compatible if they have the same arity (number of arguments)
    // and corresponding arguments are of the same type (e.g., both constants or both functions)
    if rel1.name == rel2.name && rel1.args.len() == rel2.args.len() {
        for (arg1, arg2) in rel1.args.iter().zip(rel2.args.iter()) {
            if !are_terms_compatible(arg1, arg2) {
                return false;
            }
        }
        true
    } else {
        false
    }
}

fn are_terms_compatible(term1: &Term, term2: &Term) -> bool {
    match (term1, term2) {
        (Term::ConstTerm(_), Term::ConstTerm(_)) => true,
        (Term::VarTerm(_), Term::VarTerm(_)) => true,
        (Term::FuncTerm(func1), Term::FuncTerm(func2)) => {
            if func1.name == func2.name && func1.args.len() == func2.args.len() {
                for (arg1, arg2) in func1.args.iter().zip(func2.args.iter()) {
                    if !are_terms_compatible(arg1, arg2) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
        _ => false,
    }
}

fn get_all_relations(description: &Description) -> Vec<&Relation> {
    let mut relations = Vec::new();
    for clause in &description.clauses {
        if let Clause::SentenceClause(sentence) = clause {
            if let Sentence::RelSentence(relation) = sentence {
                relations.push(relation);
            }
        }
    }
    relations
}

fn crossover(parent1: &Description, parent2: &Description) -> Description {
    let mut rng = thread_rng();

    let parent1_relations = get_all_relations(parent1);
    let parent2_relations = get_all_relations(parent2);

    if parent1_relations.is_empty() || parent2_relations.is_empty() {
        return parent1.clone(); // No crossover possible
    }

    // Select a random relation from parent1
    let rel1 = parent1_relations.choose(&mut rng).unwrap();

    // Find compatible relations in parent2
    let compatible_relations: Vec<&Relation> = parent2_relations
        .iter()
        .filter(|&&rel2| are_relations_compatible(rel1, rel2))
        .cloned()
        .collect();

    if compatible_relations.is_empty() {
        return parent1.clone(); // No compatible crossover possible
    }

    // Select a random compatible relation from parent2
    let rel2 = compatible_relations.choose(&mut rng).unwrap();

    // Create a child by swapping the compatible relations
    let mut child = parent1.clone();
    replace_relation(&mut child, rel1, rel2);

    child
}

fn replace_relation(description: &mut Description, target: &Relation, replacement: &Relation) {
    for clause in &mut description.clauses {
        if let Clause::SentenceClause(sentence) = clause {
            if let Sentence::RelSentence(relation) = sentence {
                if relation == target {
                    *relation = replacement.clone();
                    return;
                }
            }
        }
    }
}

fn fitness(program_text: &str) -> usize {
    if parse_gdl(program_text).is_some() {
        1 // Valid syntax
    } else {
        0 // Invalid syntax
    }
}

fn select_parents<'a>(
    population: &'a [Description],
    fitnesses: &[usize],
) -> (&'a Description, &'a Description) {
    let total_fitness: usize = fitnesses.iter().sum();
    let mut rng = thread_rng();

    if total_fitness == 0 {
        let idxs = rand::seq::index::sample(&mut rng, population.len(), 2).into_vec();
        (&population[idxs[0]], &population[idxs[1]])
    } else {
        let dist = rand::distributions::WeightedIndex::new(fitnesses).unwrap();
        let parent1 = &population[dist.sample(&mut rng)];
        let parent2 = &population[dist.sample(&mut rng)];
        (parent1, parent2)
    }
}

fn genetic_algorithm(
    population: Vec<Description>,
    generations: usize,
) -> Vec<Description> {
    let mut population = population;
    for generation in 0..generations {
        let fitnesses: Vec<usize> = population
            .iter()
            .map(|individual| fitness(&tree_to_gdl(individual)))
            .collect();

        // Elitism
        let max_fitness = *fitnesses.iter().max().unwrap_or(&0);
        let elite = if max_fitness > 0 {
            let elite_index = fitnesses
                .iter()
                .position(|&f| f == max_fitness)
                .unwrap();
            population[elite_index].clone()
        } else {
            population
                .choose(&mut rand::thread_rng())
                .unwrap()
                .clone()
        };

        let mut new_population = vec![elite];

        while new_population.len() < population.len() {
            let (parent1, parent2) = select_parents(&population, &fitnesses);
            let child = crossover(parent1, parent2);
            new_population.push(child);
        }

        population = new_population;
        println!("Generation {} complete.", generation + 1);
    }
    population
}

fn main() {
    // Load parent GDL programs
    let program1_text = match fs::read_to_string("tictactoe") {
        Ok(text) => text,
        Err(e) => {
            println!("Error reading tictactoe: {}", e);
            return;
        }
    };

    let program2_text = match fs::read_to_string("connect4") {
        Ok(text) => text,
        Err(e) => {
            println!("Error reading connect4: {}", e);
            return;
        }
    };

    // Parse programs into ASTs
    let program1_ast = match parse_gdl(&program1_text) {
        Some(ast) => ast,
        None => {
            println!("Error parsing tictactoe");
            return;
        }
    };

    let program2_ast = match parse_gdl(&program2_text) {
        Some(ast) => ast,
        None => {
            println!("Error parsing connect4");
            return;
        }
    };

    // Initial population
    let population = vec![program1_ast, program2_ast];

    // Run the genetic algorithm
    let evolved_population = genetic_algorithm(population, 10);

    // Save the evolved programs
    let output_dir = "evolved_programs";
    if !Path::new(output_dir).exists() {
        fs::create_dir(output_dir).expect("Failed to create output directory");
    }

    for (idx, individual) in evolved_population.iter().enumerate() {
        let program_text = tree_to_gdl(individual);
        let file_path = format!("{}/evolved_program_{}", output_dir, idx);
        fs::write(file_path, program_text).expect("Failed to write evolved program");
    }
}