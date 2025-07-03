count=1
while [ $count -le 100 ]; do
  mojo main.mojo >> outintimes.txt
  ((count++))
done
