#!/bin/bash
# Fix imports in all Polish scripts

for file in scripts/01_polish/*.py; do
    if [[ "$file" == *"_fix"* ]]; then
        continue
    fi
    
    echo "Fixing $(basename $file)..."
    
    # Replace old path with new
    sed -i '' 's|sys\.path\.insert(0, str(Path(__file__)\.parent\.parent))|PROJECT_ROOT = Path(__file__).parent.parent.parent\nsys.path.insert(0, str(PROJECT_ROOT))|g' "$file"
    
    # Replace project_root = Path() with comment
    sed -i '' 's|project_root = Path(__file__)\.parent\.parent|# PROJECT_ROOT already defined above|g' "$file"
    
    # Replace project_root references with PROJECT_ROOT
    sed -i '' 's/\bproject_root\b/PROJECT_ROOT/g' "$file"
    
done

echo "Done!"
