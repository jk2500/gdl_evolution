use gdl_parser::{parse, Clause, Description, Relation, Sentence, Term, Rule};
use rand::prelude::*;
use std::fs;
use std::path::Path;



fn generate_diff_description(evolved: &Description, parent1: &Description, parent2: &Description) -> String {
    let evolved_relations = get_all_relations(evolved);
    let parent1_relations = get_all_relations(parent1);
    let parent2_relations = get_all_relations(parent2);

    let mut description = String::from("\n\n;; Evolutionary Changes Description:\n");
    
    // Check for relations from parent1
    description.push_str(";; Inherited from Tic-tac-toe:\n");
    for rel in &evolved_relations {
        if parent1_relations.contains(rel) {
            description.push_str(&format!(";; - {}\n", rel));
        }
    }

    // Check for relations from parent2
    description.push_str(";; Inherited from Connect4:\n");
    for rel in &evolved_relations {
        if parent2_relations.contains(rel) {
            description.push_str(&format!(";; - {}\n", rel));
        }
    }

    // Check for new relations not in either parent
    description.push_str(";; New evolved relations:\n");
    for rel in &evolved_relations {
        if !parent1_relations.contains(rel) && !parent2_relations.contains(rel) {
            description.push_str(&format!(";; - {}\n", rel));
        }
    }

    description
}


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

    match clause {
        // Handle SentenceClause type
        Clause::SentenceClause(sentence) => {
            if let Sentence::RelSentence(relation) = sentence {
                if are_relations_compatible(relation, target_relation) {
                    compatible_subtrees.push(relation);
                }
            }
        }
        // Handle RuleClause type
        Clause::RuleClause(rule) => {
            // Extract the Relation from the rule head Sentence
            if let Sentence::RelSentence(relation) = &rule.head {
                if are_relations_compatible(relation, target_relation) {
                    compatible_subtrees.push(relation);
                }
            }
        }
        _ => {} // Ignore other cases if any
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
    let mut rng = thread_rng(); // Random number generator
    let crossover_rate = 0.10; // Define the crossover rate (10%)
    let mut child = parent1.clone(); // Start by cloning parent1 to create the child

    // Collect all clauses from each parent
    let parent1_clauses: Vec<_> = parent1.clauses.iter().collect();
    let parent2_clauses: Vec<_> = parent2.clauses.iter().collect();

    // If either parent has no clauses, return the child as a clone of parent1
    if parent1_clauses.is_empty() || parent2_clauses.is_empty() {
        return child;
    }

    // Iterate over each clause in parent1
    for (idx, clause1) in parent1_clauses.iter().enumerate() {
        // Perform crossover with a probability defined by `crossover_rate`
        if rng.gen::<f64>() < crossover_rate {
            // Extract the target relation from the current clause in parent1
            let target_relation = match clause1 {
                // If the clause is a SentenceClause with a RelSentence, extract the Relation
                Clause::SentenceClause(Sentence::RelSentence(rel)) => Some(rel),

                // If the clause is a SentenceClause with a PropSentence, skip
                Clause::SentenceClause(Sentence::PropSentence(_)) => None,

                // If the clause is a RuleClause, extract the Relation from the rule's head
                Clause::RuleClause(rule) => {
                    if let Sentence::RelSentence(rel) = &rule.head {
                        Some(rel)
                    } else {
                        None
                    }
                }
            };

            // If a target_relation was successfully extracted
            if let Some(target_relation) = target_relation {
                let mut compatible_subtrees = Vec::new(); // Vector to store compatible subtrees

                // Check all clauses in parent2 for compatibility with the target_relation
                for clause2 in &parent2_clauses {
                    compatible_subtrees.extend(find_compatible_subtrees(clause2, target_relation));
                }

                // Remove any subtrees that already exist in the child to avoid duplicates
                compatible_subtrees.retain(|&subtree| {
                    !child.clauses.iter().any(|child_clause| {
                        match child_clause {
                            // Check if the child clause matches the subtree
                            Clause::SentenceClause(Sentence::RelSentence(child_rel)) => child_rel == subtree,
                            Clause::SentenceClause(Sentence::PropSentence(_)) => false, // No compatibility with PropSentence
                            Clause::RuleClause(child_rule) => {
                                if let Sentence::RelSentence(child_rel) = &child_rule.head {
                                    child_rel == subtree
                                } else {
                                    false
                                }
                            }
                        }
                    })
                });

                // If compatible subtrees are found, randomly select one to replace a clause in the child
                if !compatible_subtrees.is_empty() {
                    if let Some(&replacement_relation) = compatible_subtrees.choose(&mut rng) {
                        // Find the clause in parent2 that contains the selected replacement relation
                        let replacement_clause = parent2.clauses.iter().find(|clause| {
                            match clause {
                                // Match SentenceClause with the replacement relation
                                Clause::SentenceClause(Sentence::RelSentence(rel)) => rel == replacement_relation,
                                Clause::SentenceClause(Sentence::PropSentence(_)) => false, // No match with PropSentence
                                Clause::RuleClause(rule) => {
                                    if let Sentence::RelSentence(rel) = &rule.head {
                                        rel == replacement_relation
                                    } else {
                                        false
                                    }
                                }
                            }
                        });

                        // Replace the corresponding clause in the child with the selected clause
                        if let Some(replacement_clause) = replacement_clause {
                            if let Some(child_clause) = child.clauses.get_mut(idx) {
                                *child_clause = replacement_clause.clone();
                            }
                        }
                    }
                }
            }
        }
    }

    // Return the resulting child after crossover
    child
}


// Update CrossoverStats to track subtree information
#[derive(Debug)]
struct CrossoverStats {
    total_subtrees: usize,
    crossovers_performed: usize,
    crossover_rate: f64,
    average_subtree_size: f64,
}



fn crossover_with_stats(parent1: &Description, parent2: &Description) -> (Description, CrossoverStats) {
    let mut rng = thread_rng(); // Random number generator
    let crossover_rate = 0.10; // 10% crossover rate
    let mut child = parent1.clone(); // Start with a clone of parent1
    let mut crossovers_performed = 0; // Track the number of crossovers
    let mut total_subtree_size = 0; // Track the total size of subtrees replaced

    let parent1_clauses: Vec<_> = parent1.clauses.iter().collect(); // Collect clauses from parent1
    let parent2_clauses: Vec<_> = parent2.clauses.iter().collect(); // Collect clauses from parent2
    let total_subtrees = parent1_clauses.len(); // Total subtrees in parent1

    if !parent1_clauses.is_empty() && !parent2_clauses.is_empty() {
        for (idx, clause1) in parent1_clauses.iter().enumerate() {
            // Perform crossover based on the crossover_rate
            if rng.gen::<f64>() < crossover_rate {
                let target_relation = match clause1 {
                    // Extract the relation from a RelSentence
                    Clause::SentenceClause(Sentence::RelSentence(rel)) => Some(rel),

                    // Skip PropSentence as it doesn't contribute to subtrees
                    Clause::SentenceClause(Sentence::PropSentence(_)) => None,

                    // Extract the relation from the head of a RuleClause if it exists
                    Clause::RuleClause(rule) => {
                        if let Sentence::RelSentence(rel) = &rule.head {
                            Some(rel)
                        } else {
                            None
                        }
                    }
                };

                if let Some(target_relation) = target_relation {
                    let mut compatible_subtrees = Vec::new();

                    // Find compatible subtrees in parent2
                    for clause2 in &parent2_clauses {
                        compatible_subtrees.extend(find_compatible_subtrees(clause2, target_relation));
                    }

                    // Remove subtrees already present in the child
                    compatible_subtrees.retain(|&subtree| {
                        !child.clauses.iter().any(|child_clause| {
                            match child_clause {
                                // Check for matching subtrees in RelSentences
                                Clause::SentenceClause(Sentence::RelSentence(child_rel)) => child_rel == subtree,
                                
                                // Ignore PropSentences and other clauses
                                _ => false,
                            }
                        })
                    });

                    // Replace clause in child with a randomly selected compatible subtree
                    if !compatible_subtrees.is_empty() {
                        if let Some(&replacement) = compatible_subtrees.choose(&mut rng) {
                            if let Some(clause) = child.clauses.get_mut(idx) {
                                *clause = Clause::SentenceClause(Sentence::RelSentence(replacement.clone()));
                                crossovers_performed += 1;
                                total_subtree_size += replacement.args.len();
                            }
                        }
                    }
                }
            }
        }
    }

    // Calculate the average size of the replaced subtrees
    let average_subtree_size = if crossovers_performed > 0 {
        total_subtree_size as f64 / crossovers_performed as f64
    } else {
        0.0
    };

    // Generate the stats for this crossover operation
    let stats = CrossoverStats {
        total_subtrees,
        crossovers_performed,
        crossover_rate,
        average_subtree_size,
    };

    // Return the resulting child and the crossover stats
    (child, stats)
}


fn genetic_algorithm(
    population: Vec<Description>,
    generations: usize,
) -> Vec<Description> {
    let mut population = population;
    let mut generation_stats = Vec::new();

    for generation in 0..generations {
        let fitnesses: Vec<usize> = population
            .iter()
            .map(|individual| fitness(&tree_to_gdl(individual)))
            .collect();

        // Elitism - keep the best performing individual
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
        let mut generation_crossover_stats = Vec::new();

        while new_population.len() < population.len() {
            let (parent1, parent2) = select_parents(&population, &fitnesses);
            let (child, stats) = crossover_with_stats(parent1, parent2);
            generation_crossover_stats.push(stats);
            new_population.push(child);
        }

        // Print statistics for this generation
        let avg_crossovers = generation_crossover_stats.iter()
            .map(|s| s.crossovers_performed)
            .sum::<usize>() as f64 / generation_crossover_stats.len() as f64;

        println!("Generation {} complete:", generation + 1);
        println!("  Average crossovers per individual: {:.2}", avg_crossovers);
        println!("  Total individuals: {}", generation_crossover_stats.len()+1);

        generation_stats.push(generation_crossover_stats);
        population = new_population;
    }

    population
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

fn main() {


        

        let num_programs = 5;  // Add this line to control number of outputs
        
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

            Some(ast) => {
                println!("Parsed AST for tictactoe:\n{:?}", ast);
                ast
            },
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
    
        // Initial population - create num_programs copies alternating between the two source programs
        let mut population = Vec::with_capacity(num_programs);
        for i in 0..num_programs {
            if i % 2 == 0 {
                population.push(program1_ast.clone());
            } else {
                population.push(program2_ast.clone());
            }
        }

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

    for (idx, individual) in evolved_population.iter().enumerate() {
        let mut program_text = tree_to_gdl(individual);
        let diff_description = generate_diff_description(individual, &program1_ast, &program2_ast);
        program_text.push_str(&diff_description);
        
        let file_path = format!("{}/evolved_program_{}", output_dir, idx);
        fs::write(file_path, program_text).expect("Failed to write evolved program");
    }
}